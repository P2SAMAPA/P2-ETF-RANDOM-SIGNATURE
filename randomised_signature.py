import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

def lead_lag_path(price_series):
    if len(price_series) < 2:
        return None
    lead = price_series[1:].values
    lag = price_series[:-1].values
    return np.column_stack([lead, lag])

def signature_depth3(path):
    """
    Compute truncated signature of a 2D path up to depth 3.
    Returns 14‑dim vector.
    """
    if path is None or len(path) < 2:
        return None
    inc = np.diff(path, axis=0)   # (L, 2)
    L = inc.shape[0]
    # Depth 1
    sig1 = np.sum(inc, axis=0)          # (2,)
    # Depth 2
    sig2 = np.zeros(4)
    idx = 0
    for i in range(2):
        for j in range(2):
            total = 0.0
            for s in range(L):
                prefix = np.sum(inc[:s+1, i])
                total += prefix * inc[s, j]
            sig2[idx] = total
            idx += 1
    # Depth 3
    sig3 = np.zeros(8)
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0.0
                for s2 in range(L):
                    double = 0.0
                    for s1 in range(s2+1):
                        prefix = np.sum(inc[:s1+1, i])
                        double += prefix * inc[s1, j]
                    total += double * inc[s2, k]
                sig3[idx] = total
                idx += 1
    return np.concatenate([sig1, sig2, sig3])   # (14,)

def generate_random_projection(d_in, d_out, seed=42):
    np.random.seed(seed)
    # Gaussian random matrix with scale 1/sqrt(d_out) for JL property
    proj = np.random.randn(d_out, d_in) / np.sqrt(d_out)
    return proj

def project_signature(sig, proj):
    return proj @ sig

def train_ridge_on_random_signature(returns_df, ticker, window, sig_depth, random_dim, ridge_alpha):
    """
    For a single ETF, build training samples from rolling windows,
    compute full signature, project randomly, then train ridge.
    Returns (model, scaler, projection_matrix).
    """
    price_series = returns_df[ticker]  # actually we need price, not return. So we need prices.
    # We'll use price matrix from outside.
    # Better: pass the price series to this function.
    pass

# We'll restructure: trainer will handle the loop.
