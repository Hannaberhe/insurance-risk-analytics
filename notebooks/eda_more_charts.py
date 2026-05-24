import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/insurance_data.csv', sep='|', low_memory=False)
df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
df['Loss_Ratio'] = df['TotalClaims'] / df['TotalPremium']

# Chart 1: Histogram of TotalPremium
fig, ax = plt.subplots(figsize=(10, 5))
df['TotalPremium'].dropna().hist(bins=50, ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Distribution of Total Premium')
ax.set_xlabel('Premium (R)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('reports/histogram_premium.png', dpi=150)
print("Chart 1 saved!")

# Chart 2: Histogram of TotalClaims
fig, ax = plt.subplots(figsize=(10, 5))
df[df['TotalClaims'] > 0]['TotalClaims'].hist(bins=50, ax=ax, color='coral', edgecolor='black')
ax.set_title('Distribution of Total Claims (claims > 0)')
ax.set_xlabel('Claims (R)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig('reports/histogram_claims.png', dpi=150)
print("Chart 2 saved!")

# Chart 3: Boxplot of TotalClaims by Province
fig, ax = plt.subplots(figsize=(12, 6))
top_provs = df['Province'].value_counts().head(5).index
df_filtered = df[df['Province'].isin(top_provs)]
df_filtered.boxplot('TotalClaims', by='Province', ax=ax)
ax.set_title('Claims Distribution by Top 5 Provinces')
ax.set_xlabel('Province')
ax.set_ylabel('Claims (R)')
plt.tight_layout()
plt.savefig('reports/boxplot_claims_province.png', dpi=150)
print("Chart 3 saved!")

# Chart 4: Correlation matrix
num_cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']
num_df = df[num_cols].dropna()
corr = num_df.corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
ax.set_title('Correlation Matrix')
plt.tight_layout()
plt.savefig('reports/correlation_matrix.png', dpi=150)
print("Chart 4 saved!")

# Chart 5: Loss Ratio by Vehicle Type
fig, ax = plt.subplots(figsize=(12, 6))
vehicle_lr = df.groupby('VehicleType')['Loss_Ratio'].mean().sort_values().tail(10)
vehicle_lr.plot(kind='barh', ax=ax, color='teal')
ax.set_title('Loss Ratio by Vehicle Type (Top 10)')
ax.set_xlabel('Loss Ratio')
plt.tight_layout()
plt.savefig('reports/loss_ratio_vehicle.png', dpi=150)
print("Chart 5 saved!")

print("\nAll 5 charts saved in reports/")
