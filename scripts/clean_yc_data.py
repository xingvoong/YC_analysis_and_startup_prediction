import pandas as pd
from pathlib import Path
from datetime import date

RAW_PATH = Path("../data/raw/yc_companies_raw.csv")
CLEAN_PATH = Path("../data/cleaned/yc_companies_clean.csv")


def parse_batch_year(batch):
    if not isinstance(batch, str) or len(batch) < 3:
        return None

    year_part = batch[-2:]
    if not year_part.isdigit():
        return None

    year = int(year_part)
    return 2000 + year if year <= 29 else 1900 + year


def parse_batch_season(batch):
    if isinstance(batch, str) and batch:
        return batch[0]
    return None


def normalize_company_size(size):
    """
    company_size_num:
    Last-known team size reported to YC.
    Not real-time. Not guaranteed current.
    """
    if pd.isna(size):
        return None
    try:
        return int(size)
    except (ValueError, TypeError):
        return None


def bucket_company_size(size):
    """
    company_size_bucket:
    Stable categorical approximation of company maturity.
    Safer than raw size for analysis / ML.
    """
    if size is None:
        return None
    if size == 1:
        return "solo"
    if size <= 10:
        return "small"
    if size <= 50:
        return "early"
    if size <= 200:
        return "scaling"
    return "large"


print("Loading raw YC data...")
df = pd.read_csv(RAW_PATH)
print(f"Rows loaded: {len(df)}")

print("Cleaning data...")

df["batch_year"] = df["batch"].apply(parse_batch_year)
df["batch_season"] = df["batch"].apply(parse_batch_season)

df["company_size_num"] = df["company_size"].apply(normalize_company_size)
df["company_size_bucket"] = df["company_size_num"].apply(bucket_company_size)

df["as_of_date"] = date.today().isoformat()

# Keep only valid YC batches
df = df[df["batch_year"].notna()]

print(f"Rows after cleaning: {len(df)}")
print("Company size bucket distribution:")
print(df["company_size_bucket"].value_counts(dropna=False))

CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_PATH, index=False)

print(f"Saved cleaned data to {CLEAN_PATH}")
print("Batch year range:", df["batch_year"].min(), "→", df["batch_year"].max())
