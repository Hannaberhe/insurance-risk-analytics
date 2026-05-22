import pandas as pd
import numpy as np
from scipy import stats

# Load data
df = pd.read_csv('data/insurance_data.csv', sep='|', low_memory=False)
df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

print("Data loaded:", len(df), "rows")
print("\n" + "="*50)
print("HYPOTHESIS TESTING RESULTS")
print("="*50)

# Test 1: Provinces (ANOVA)
provinces = df['Province'].value_counts().head(3).index
groups = []
for p in provinces:
    group = df[df['Province'] == p]['TotalClaims'].dropna()
    if len(group) > 30:
        groups.append(group)

if len(groups) >= 2:
    f_stat, p_val = stats.f_oneway(*groups)
    print("\n1. Provinces risk difference")
    print("   Test: ANOVA")
    print("   P-value:", round(p_val, 4))
    if p_val < 0.05:
        print("   Result: REJECT - Provinces have different risk levels")
        province_avg = df.groupby('Province')['TotalClaims'].mean()
        print("   Highest:", province_avg.idxmax(), round(province_avg.max(), 2))
        print("   Lowest:", province_avg.idxmin(), round(province_avg.min(), 2))
    else:
        print("   Result: No significant difference found")

# Test 2: Zip codes (T-test)
top_zips = df['PostalCode'].value_counts().head(2).index
zip1 = df[df['PostalCode'] == top_zips[0]]['TotalClaims'].dropna()
zip2 = df[df['PostalCode'] == top_zips[1]]['TotalClaims'].dropna()

if len(zip1) > 30 and len(zip2) > 30:
    t_stat, p_val = stats.ttest_ind(zip1, zip2)
    print("\n2. Zip code risk difference")
    print("   Test: T-test")
    print("   P-value:", round(p_val, 4))
    if p_val < 0.05:
        print("   Result: REJECT - Zip codes have different risk")
    else:
        print("   Result: No significant difference")

# Test 3: Margin by zip (T-test)
zip1_m = df[df['PostalCode'] == top_zips[0]]['Margin'].dropna()
zip2_m = df[df['PostalCode'] == top_zips[1]]['Margin'].dropna()

if len(zip1_m) > 30 and len(zip2_m) > 30:
    t_stat, p_val = stats.ttest_ind(zip1_m, zip2_m)
    print("\n3. Margin difference by zip code")
    print("   Test: T-test")
    print("   P-value:", round(p_val, 4))
    if p_val < 0.05:
        print("   Result: REJECT - Margins differ between zips")
    else:
        print("   Result: No significant difference")

# Test 4: Gender (T-test)
men = df[df['Gender'] == 'Male']['TotalClaims'].dropna()
women = df[df['Gender'] == 'Female']['TotalClaims'].dropna()

if len(men) > 30 and len(women) > 30:
    t_stat, p_val = stats.ttest_ind(men, women)
    print("\n4. Gender risk difference")
    print("   Test: T-test")
    print("   P-value:", round(p_val, 4))
    if p_val < 0.05:
        print("   Result: REJECT - Men and women have different risk")
        print("   Male avg claims:", round(men.mean(), 2))
        print("   Female avg claims:", round(women.mean(), 2))
    else:
        print("   Result: No significant difference between genders")

print("\n" + "="*50)
print("BUSINESS RECOMMENDATIONS")
print("="*50)
print("1. Focus marketing on low-claim provinces")
print("2. Adjust premiums based on zip code risk")
print("3. Consider gender-neutral pricing unless difference is large")
print("\nDone!")
