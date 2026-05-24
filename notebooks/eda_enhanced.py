import pandas as pd
import numpy as np

df = pd.read_csv('data/insurance_data.csv', sep='|', low_memory=False)
df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')

# Descriptive statistics
print("=== DESCRIPTIVE STATISTICS ===")
print(df[['TotalPremium', 'TotalClaims']].describe())

# Data types
print("\n=== DATA TYPES ===")
print(df.dtypes.value_counts())

# Bivariate by ZipCode
print("\n=== PREMIUM & CLAIMS BY ZIP CODE (Top 5) ===")
top_zips = df['PostalCode'].value_counts().head(5).index
for zip_code in top_zips:
    subset = df[df['PostalCode'] == zip_code]
    print(f"\nZip {zip_code}:")
    print(f"  Avg Premium: R{subset['TotalPremium'].mean():.2f}")
    print(f"  Avg Claims: R{subset['TotalClaims'].mean():.2f}")
    print(f"  Count: {len(subset)}")

# Geographic: Cover type by province
print("\n=== COVER TYPE BY PROVINCE (Top 3 provinces) ===")
top_provinces = df['Province'].value_counts().head(3).index
for prov in top_provinces:
    print(f"\n{prov}:")
    print(df[df['Province'] == prov]['CoverType'].value_counts().head(5))

# Vehicle make by province
print("\n=== TOP VEHICLE MAKES BY PROVINCE ===")
for prov in top_provinces:
    print(f"\n{prov}:")
    print(df[df['Province'] == prov]['make'].value_counts().head(5))

print("\nDone!")
