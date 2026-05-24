# Insurance Risk Analytics

AlphaCare Insurance Solutions - Predictive Analytics Project

## Setup
pip install -r requirements.txt

## Project Structure
- data/ - Insurance dataset (DVC tracked)
- notebooks/ - Analysis notebooks
- src/ - Reusable Python modules
- reports/ - Final report

## DVC Setup
1. Install DVC: `pip install dvc`
2. Pull data: `dvc pull`
3. Reproduce pipeline: `dvc repro`

## Data Versions
- `data/insurance_data.csv` - Raw data (pipe-delimited)
- `data/insurance_data_clean.csv` - Cleaned data (numeric types, no nulls in key columns)
