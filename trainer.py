import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import config
import data_manager
from randomised_signature import lead_lag_path, signature_depth3, generate_random_projection, project_signature
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}
    today = datetime.now().strftime("%Y-%m-%d")

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (Randomised Signature) ===")
        prices = data_manager.prepare_price_matrix(df, tickers)
        if prices.empty or len(prices) < max(config.WINDOWS) + config.WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Pre‑compute random projection matrix (same for all ETFs and windows)
        d_in = 14   # signature depth 3
        d_out = config.RANDOM_DIM
        proj = generate_random_projection(d_in, d_out, seed=42)

        best_per_etf = {}
        window_results = {}

        for win in config.WINDOWS:
            if len(prices) < win + config.WINDOW + 1:
                print(f"  Skipping window {win}d (insufficient data)")
                continue
            print(f"  Processing window {win}d...")
            etf_pred = {}
            for etf in tickers:
                if etf not in prices.columns:
                    continue
                # Build training data using rolling windows of size `config.WINDOW`
                price_series = prices[etf].dropna()
                if len(price_series) < win + config.WINDOW + 1:
                    continue
                # Use the last `win` days as the training period? No, we need to create a supervised dataset from the last `win` days.
                # Actually, we use the last `win` days as the data for training, sliding a window of size `config.WINDOW` inside it.
                # Simpler: For each ETF, we use the whole `win` period to generate signatures and train the model.
                # We'll slide over the most recent `win` days: for each i in [WINDOW, win-1], create signature of the last WINDOW days up to i.
                # But that would use data only from the training window; to predict the next day after the window, we need to use the last WINDOW days of the training window.
                # We'll implement a walk‑forward inside the window: for each i in [WINDOW, win-1], train on data from i-WINDOW to i, predict next day.
                # This is similar to the other engines.
                # However, to keep it simple and fast, we train on the entire `win` period and then predict using the most recent WINDOW days.
                # That's what we'll do: use the last `win` days as the training set, then predict the next day.
                # Build features: for each day from WINDOW to win-1, signature of the last WINDOW days of prices.
                X, y = [], []
                for i in range(config.WINDOW, win):
                    window_prices = price_series.iloc[i-config.WINDOW:i]
                    path = lead_lag_path(window_prices)
                    if path is None:
                        continue
                    sig = signature_depth3(path)
                    if sig is None:
                        continue
                    proj_sig = project_signature(sig, proj)
                    X.append(proj_sig)
                    # target: next day return
                    ret = np.log(price_series.iloc[i+1] / price_series.iloc[i]) if i+1 < len(price_series) else np.nan
                    if np.isnan(ret):
                        continue
                    y.append(ret)
                if len(X) < 10:
                    continue
                X = np.array(X)
                y = np.array(y)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model = Ridge(alpha=config.RIDGE_ALPHA)
                model.fit(X_scaled, y)
                # Predict for the most recent window (last WINDOW days of the training window)
                last_window = price_series.iloc[-config.WINDOW:]
                last_path = lead_lag_path(last_window)
                if last_path is None:
                    continue
                last_sig = signature_depth3(last_path)
                if last_sig is None:
                    continue
                last_proj = project_signature(last_sig, proj).reshape(1, -1)
                last_scaled = scaler.transform(last_proj)
                pred = model.predict(last_scaled)[0]
                etf_pred[etf] = pred
            window_results[win] = etf_pred
            for etf, pred in etf_pred.items():
                if etf not in best_per_etf or pred > best_per_etf[etf][0]:
                    best_per_etf[etf] = (pred, win)

        if not best_per_etf:
            print("  No valid predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Store full scores for all ETFs
        full_scores = {ticker: {"score": score, "best_window": win} for ticker, (score, win) in best_per_etf.items()}
        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1][0], reverse=True)
        top_etfs = [{"ticker": ticker, "pred_return": float(score), "best_window": win} for ticker, (score, win) in sorted_etfs[:config.TOP_N]]

        print(f"  Top 3 ETFs: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
            "window_results": window_results,
            "run_date": today
        }

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/random_sig_{today}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": today, "universes": all_results}, f, indent=2)

    import push_results
    push_results.push_daily_result(local_path)
    print("\n=== Randomised Signature Engine complete ===")

if __name__ == "__main__":
    main()
