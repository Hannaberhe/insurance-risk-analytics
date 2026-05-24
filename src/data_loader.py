"""Data loading with error handling."""
import pandas as pd
import os

def load_data(filepath='data/insurance_data.csv'):
    """Load insurance data with validation."""
    
    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    try:
        # Try pipe separator first
        df = pd.read_csv(filepath, sep='|', low_memory=False)
        
        # Validate required columns
        required = ['TotalPremium', 'TotalClaims']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert to numeric
        df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
        df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
        
        print(f"Data loaded successfully: {len(df)} rows")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
