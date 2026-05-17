import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import config

def load_master_data():
    path = hf_hub_download(repo_id=config.DATA_REPO, filename="master_data.parquet", repo_type="dataset", token=config.HF_TOKEN)
    df = pd.read_parquet(path)
    if df.index.name != 'date':
        df.index.name = 'date'
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    return df

def prepare_price_matrix(df, universe_tickers):
    prices = pd.DataFrame(index=df.index)
    for ticker in universe_tickers:
        if ticker in df.columns:
            prices[ticker] = df[ticker]
    prices = prices.dropna(how='all')
    return prices
