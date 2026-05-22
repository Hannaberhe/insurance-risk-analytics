"""Data loading utilities."""
import pandas as pd

def load_data(filepath='data/insurance_data.csv'):
    """Load insurance data with pipe separator."""
    df = pd.read_csv(filepath, sep='|', low_memory=False)
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
    return df
