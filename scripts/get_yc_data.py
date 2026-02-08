import requests
import pandas as pd
from pathlib import Path
from datetime import date

# Output path
OUTPUT_PATH = Path("../data/raw/yc_companies_raw.csv")

# YC OSS dataset (full historical)
URL = "https://yc-oss.github.io/api/companies/all.json"

print("Downloading YC companies dataset...")

response = requests.get(URL, timeout=30)
response.raise_for_status()

companies = response.json()
print(f"Loaded {len(companies)} companies")

# Normalize JSON into flat table
df = pd.json_normalize(companies)

# Rename key fields for consistency
df = df.rename(columns={
    "name": "company",
    "slug": "slug",
    "batch": "batch",
    "one_liner": "description",
    "long_description": "long_description",
    "team_size": "company_size",
    "all_locations": "locations",
    "website": "website",
    "status": "status"
})

# Keep only relevant columns (others still exist if needed)
keep_cols = [
    "company",
    "slug",
    "batch",
    "status",
    "company_size",
    "locations",
    "description",
    "long_description",
    "website"
]

df = df[[c for c in keep_cols if c in df.columns]]

# Add snapshot metadata
df["as_of_date"] = date.today().isoformat()

# Write output
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
print("Columns:", list(df.columns))
