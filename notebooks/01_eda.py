import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

# Load data with pipe separator
df = pd.read_csv('data/insurance_data.csv', sep='|', low_memory=False)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# Data Quality
missing = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
print("\nMissing Values:")
print(missing_pct[missing_pct > 0].head(20))

# Key metrics
df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
df['Loss_Ratio'] = df['TotalClaims'] / df['TotalPremium']
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

print(f"\nOverall Loss Ratio: {df['TotalClaims'].sum() / df['TotalPremium'].sum():.4f}")
print(f"Average Margin: R{df['Margin'].mean():.2f}")

# Chart 1: Loss Ratio by Province
fig, ax = plt.subplots(figsize=(10, 6))
province_lr = df.groupby('Province')['Loss_Ratio'].mean().sort_values()
province_lr.plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Average Loss Ratio by Province')
ax.set_xlabel('Loss Ratio')
plt.tight_layout()
plt.savefig('reports/loss_ratio_by_province.png', dpi=150)
print("Chart 1 saved!")

# Chart 2: Premium vs Claims
fig, ax = plt.subplots(figsize=(10, 6))
sample = df.sample(5000) if len(df) > 5000 else df
ax.scatter(sample['TotalPremium'], sample['TotalClaims'], alpha=0.3)
ax.set_xlabel('Total Premium')
ax.set_ylabel('Total Claims')
ax.set_title('Premium vs Claims')
plt.tight_layout()
plt.savefig('reports/premium_vs_claims.png', dpi=150)
print("Chart 2 saved!")

# Chart 3: Claims by Province
fig, ax = plt.subplots(figsize=(10, 6))
province_claims = df.groupby('Province')['TotalClaims'].mean().sort_values()
province_claims.plot(kind='barh', ax=ax, color='coral')
ax.set_title('Average Claims by Province')
ax.set_xlabel('Average Claim Amount (R)')
plt.tight_layout()
plt.savefig('reports/claims_by_province.png', dpi=150)
print("Chart 3 saved!")

print("\nEDA complete!")
