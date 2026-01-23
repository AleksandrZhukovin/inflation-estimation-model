import pandas as pd
import eurostat

print("--- 1. Fetching Eurostat Data ---")
# Fetch Monthly HICP (Inflation)
filter_pars = {"geo": ["LV", "LT"], "coicop": ["CP00"], "unit": ["I15"], "freq": ["M"]}
df = eurostat.get_data_df("prc_hicp_midx", filter_pars=filter_pars)

# DEBUG: Print the raw columns to see what we are dealing with
print(f"Raw Data Shape: {df.shape}")
print("First 10 Columns found:", df.columns[:10].tolist())

# --- 2. Smart Column Identification ---
# Instead of hardcoding names, we separate "Metadata" (text) from "Dates" (time)
# Metadata columns usually contain single letters/words (like 'M', 'I15', 'LV')
# Date columns usually start with '20' or '19' (e.g., '2023M01', '2023M02')

metadata_cols = [c for c in df.columns if not c[0].isdigit()]
date_cols = [c for c in df.columns if c[0].isdigit()]

print(f"Identified {len(metadata_cols)} metadata columns: {metadata_cols}")
print(f"Identified {len(date_cols)} date columns (Example: {date_cols[:3]}...)")

# --- 3. Melt and Fix ---
if not date_cols:
    print(
        "ERROR: No date columns found! The Eurostat API might have changed structure."
    )
else:
    # Melt using the lists we just discovered
    df_melt = df.melt(
        id_vars=metadata_cols, value_vars=date_cols, var_name="Date", value_name="CPI"
    )

    # Remove rows with no data (NaN)
    df_melt = df_melt.dropna()

    # Clean Date Format:
    # Eurostat dates usually look like "2023M01" or "2023M1"
    # We replace 'M' with '-' and append '-01' to make it a real date
    df_melt["Date"] = df_melt["Date"].astype(str).str.replace("M", "-")
    df_melt["Date"] = pd.to_datetime(df_melt["Date"], format="%Y-%m")

    # Find the column that holds the Country Code (usually 'geo\TIME_PERIOD' or just 'geo')
    # We look for the column name that contains "geo"
    geo_col = next((c for c in metadata_cols if "geo" in c.lower()), None)

    if geo_col:
        # Pivot: Date as Index, Country Code as Columns
        df_pivot = df_melt.pivot(index="Date", columns=geo_col, values="CPI")

        # Save
        df_pivot.to_csv("euro_cpi_fixed.csv")
        print("\nSUCCESS!")
        print(df_pivot.head())
        print("Data saved to 'euro_cpi_fixed.csv'")
    else:
        print("Error: Could not find a 'geo' column to identify countries.")
