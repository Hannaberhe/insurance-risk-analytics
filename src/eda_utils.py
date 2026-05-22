"""EDA utility functions for insurance data analysis."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_insurance_data(filepath='data/insurance_data.csv'):
    """Load insurance data with pipe separator."""
    df = pd.read_csv(filepath, sep='|', low_memory=False)
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
    return df

def calculate_metrics(df):
    """Calculate loss ratio and margin."""
    df['Loss_Ratio'] = df['TotalClaims'] / df['TotalPremium']
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def missing_values_report(df):
    """Report missing values."""
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_pct = (missing / len(df) * 100).round(2)
    return missing_pct[missing_pct > 0]

def plot_loss_ratio_by_category(df, category, top_n=10):
    """Plot loss ratio by a categorical variable."""
    lr = df.groupby(category)['Loss_Ratio'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    lr.tail(top_n).plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title(f'Loss Ratio by {category}')
    ax.set_xlabel('Loss Ratio')
    plt.tight_layout()
    return fig
