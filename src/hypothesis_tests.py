"""Hypothesis testing functions for insurance risk analysis."""
import pandas as pd
import numpy as np
from scipy import stats

def load_and_prepare(filepath='data/insurance_data.csv'):
    """Load and prepare data for testing."""
    df = pd.read_csv(filepath, sep='|', low_memory=False)
    df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce')
    df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce')
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    return df

def test_provinces_anova(df):
    """H0: No risk differences across provinces (ANOVA)."""
    provinces = df['Province'].dropna().value_counts().head(4).index
    groups = [df[df['Province'] == p]['TotalClaims'].dropna() for p in provinces if len(df[df['Province'] == p]) > 30]
    f_stat, p_value = stats.f_oneway(*groups)
    return p_value

def test_zip_ttest(df):
    """H0: No risk differences between top 2 zip codes (T-test)."""
    top_zips = df['PostalCode'].value_counts().head(2).index
    zip1 = df[df['PostalCode'] == top_zips[0]]['TotalClaims'].dropna()
    zip2 = df[df['PostalCode'] == top_zips[1]]['TotalClaims'].dropna()
    t_stat, p_value = stats.ttest_ind(zip1, zip2)
    return p_value

def test_margin_zip_ttest(df):
    """H0: No margin difference between top 2 zip codes (T-test)."""
    top_zips = df['PostalCode'].value_counts().head(2).index
    zip1 = df[df['PostalCode'] == top_zips[0]]['Margin'].dropna()
    zip2 = df[df['PostalCode'] == top_zips[1]]['Margin'].dropna()
    t_stat, p_value = stats.ttest_ind(zip1, zip2)
    return p_value

def test_gender_ttest(df):
    """H0: No risk difference between Women and Men (T-test)."""
    men = df[df['Gender'] == 'Male']['TotalClaims'].dropna()
    women = df[df['Gender'] == 'Female']['TotalClaims'].dropna()
    t_stat, p_value = stats.ttest_ind(men, women)
    return p_value

def run_all_tests(df):
    """Run all hypothesis tests and return results."""
    results = []
    
    p1 = test_provinces_anova(df)
    results.append(('Provinces risk difference (ANOVA)', p1, 'Reject H0' if p1 < 0.05 else 'Fail to reject'))
    
    p2 = test_zip_ttest(df)
    results.append(('Zip codes risk difference (T-test)', p2, 'Reject H0' if p2 < 0.05 else 'Fail to reject'))
    
    p3 = test_margin_zip_ttest(df)
    results.append(('Margin zip difference (T-test)', p3, 'Reject H0' if p3 < 0.05 else 'Fail to reject'))
    
    p4 = test_gender_ttest(df)
    results.append(('Gender risk difference (T-test)', p4, 'Reject H0' if p4 < 0.05 else 'Fail to reject'))
    
    return pd.DataFrame(results, columns=['Hypothesis', 'P-value', 'Decision'])
