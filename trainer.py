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
        if prices.empty or len(prices) < config.TRAIN_WINDOW + config.WINDOW + 10:
            print("  Insufficient data")
            all_results[universe_name] = {"top_etfs": []}
            continue

        # Pre‑compute for each ETF
        etf_predictions = {}
        full_scores = {}

        # We'll generate a single random projection matrix (same for all ETFs) for reproducibility
        d_in = 14  # depth 3 signature dimension
        d_out = config.RANDOM_DIM
        proj = generate_random_projection(d_in, d_out, seed=42)

        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            # Collect training samples: for each day i (window..len-2), signature of last WINDOW days, target = next day return
            X_train = []
            y_train = []
            price_series = prices[ticker].dropna()
            for i in range(config.WINDOW, len(price_series)-1):
                window_prices = price_series.iloc[i-config.WINDOW:i]
                path = lead_lag_path(window_prices)
                if path is None:
                    continue
                sig = signature_depth3(path)
                if sig is None:
                    continue
                proj_sig = project_signature(sig, proj)
                X_train.append(proj_sig)
                # target: next day return (log return)
                ret = np.log(price_series.iloc[i+1] / price_series.iloc[i])
                y_train.append(ret)
            if len(X_train) < 50:
                continue
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # Standardise
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)
            # Ridge regression
            model = Ridge(alpha=config.RIDGE_ALPHA)
            model.fit(X_scaled, y_train)
            # Predict for the most recent window (last WINDOW days)
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
            etf_predictions[ticker] = pred

        if not etf_predictions:
            print("  No predictions")
            all_results[universe_name] = {"top_etfs": []}
            continue

        sorted_etfs = sorted(etf_predictions.items(), key=lambda x: x[1], reverse=True)
        top_etfs = []
        for ticker, pred in sorted_etfs[:config.TOP_N]:
            top_etfs.append({"ticker": ticker, "pred_return": float(pred)})
            full_scores[ticker] = float(pred)
        print(f"  Top 3 ETFs by predicted return: {[e['ticker'] for e in top_etfs]}")
        all_results[universe_name] = {
            "top_etfs": top_etfs,
            "full_scores": full_scores,
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
