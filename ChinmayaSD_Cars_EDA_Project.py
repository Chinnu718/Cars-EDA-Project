"""
Exploratory Data Analysis on a Cars Dataset
Student : Chinmaya SD (USN 1AM25CI033)
Course  : AIML - Unlox Minor Project 1

Dataset : CarDekho used-car listings (4,340 rows, 8 columns), originally published on Kaggle.
          This script reads the local file 'Car_dataset.csv' if present, otherwise it downloads the
          same file from the GitHub repository PiyushBadhe/Car-Price-Prediction-ML.

Run     : python ChinmayaSD_Cars_EDA_Project.py
Needs   : pandas, numpy, matplotlib, seaborn
Output  : prints every result to the console and saves the charts in the 'charts' folder.
"""

# =============================================================================
# 0. SETUP
# =============================================================================
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=1.0)
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 200

DATA_PATH = "Car_dataset.csv"
DATA_URL = ("https://raw.githubusercontent.com/PiyushBadhe/"
            "Car-Price-Prediction-ML/main/Car_dataset.csv")
CHART_DIR = "charts"
os.makedirs(CHART_DIR, exist_ok=True)

MAIN_COLOR = "#2f6db5"      # one consistent colour for single-series charts
ACCENT_COLOR = "#d9822b"


def save_chart(fig, name):
    """Save a figure into the charts folder and show it."""
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, name), bbox_inches="tight")
    plt.show()
    plt.close(fig)


# =============================================================================
# 1. LOAD AND INSPECT THE DATA
# =============================================================================
print("=" * 70)
print("1. DATA UNDERSTANDING")
print("=" * 70)

df_raw = pd.read_csv(DATA_PATH if os.path.exists(DATA_PATH) else DATA_URL)

print("Shape (rows, columns):", df_raw.shape)
print("\nColumn names:", list(df_raw.columns))
print("\nFirst 5 rows:")
print(df_raw.head())
print("\nData types:")
print(df_raw.dtypes)

# Separate numerical and categorical variables
numeric_cols = df_raw.select_dtypes(include="number").columns.tolist()
categorical_cols = [c for c in df_raw.columns if c not in numeric_cols]
print("\nNumerical columns  :", numeric_cols)
print("Categorical columns:", categorical_cols)

# =============================================================================
# 2. DATA QUALITY CHECKS
# =============================================================================
print("\n" + "=" * 70)
print("2. DATA QUALITY CHECKS")
print("=" * 70)

missing_counts = df_raw.isna().sum()
print("Missing values per column:")
print(missing_counts)

n_duplicates = int(df_raw.duplicated().sum())
print(f"\nDuplicate rows (all columns identical): {n_duplicates} "
      f"({n_duplicates / len(df_raw):.1%} of the data)")

print("\nCategory labels in each categorical column:")
for col in ["Fuel", "Seller_Type", "Transmission", "Owner"]:
    print(f"\n{col}:")
    print(df_raw[col].value_counts())

# Checks for whitespace / spelling inconsistencies in the car name
stray_space = int((df_raw["Car_Name"] != df_raw["Car_Name"].str.strip()).sum())
print("\nCar names with leading/trailing spaces:", stray_space)

# Logical range checks
print("Year range         :", df_raw["Year"].min(), "to", df_raw["Year"].max())
print("Price <= 0 rows    :", int((df_raw["Selling_Price"] <= 0).sum()))
print("Km_Driven <= 0 rows:", int((df_raw["Km_Driven"] <= 0).sum()))

# =============================================================================
# 3. DATA CLEANING
# =============================================================================
print("\n" + "=" * 70)
print("3. DATA CLEANING")
print("=" * 70)

df = df_raw.copy()
size_before = df.shape

# 3.1 Remove exact duplicate rows
# A used-car listing repeated with identical name, year, price, km, fuel, seller,
# transmission and owner is almost certainly the same listing scraped more than once.
df = df.drop_duplicates().reset_index(drop=True)
print(f"3.1 Duplicates removed: {size_before[0] - len(df)} rows")

# 3.2 Create a Brand column from the first word of Car_Name
df["Car_Name"] = df["Car_Name"].str.strip()
df["Brand"] = df["Car_Name"].str.split().str[0]

# Fix brand names that the first-word rule splits or glues incorrectly
df.loc[df["Car_Name"].str.startswith("Land Rover"), "Brand"] = "Land Rover"
df.loc[df["Car_Name"].str.startswith("OpelCorsa"), "Brand"] = "Opel"
print("3.2 Brand column created, unique brands:", df["Brand"].nunique())

# 3.3 Make the Owner labels shorter and consistent
owner_map = {
    "First Owner": "First",
    "Second Owner": "Second",
    "Third Owner": "Third",
    "Fourth & Above Owner": "Fourth+",
    "Test Drive Car": "Test Drive",
}
df["Owner"] = df["Owner"].map(owner_map)
print("3.3 Owner labels standardised:", df["Owner"].unique().tolist())

# 3.4 Feature engineering
# The newest model year in the data is 2020, so listings are treated as of 2020.
reference_year = int(df["Year"].max())
df["Car_Age"] = reference_year - df["Year"]
df["Price_Lakh"] = df["Selling_Price"] / 100000      # 1 lakh = 100,000 INR
print(f"3.4 Car_Age = {reference_year} - Year, Price_Lakh = Selling_Price / 100000")

# 3.5 Convert text columns to the category data type (correct unsuitable types)
for col in ["Brand", "Fuel", "Seller_Type", "Transmission", "Owner"]:
    df[col] = df[col].astype("category")

# 3.6 Outlier detection with the IQR rule (report only, do not blindly delete)
print("\n3.6 Outlier check (IQR rule: below Q1 - 1.5*IQR or above Q3 + 1.5*IQR)")
outlier_table = []
for col in ["Selling_Price", "Km_Driven", "Car_Age"]:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((df[col] < low) | (df[col] > high)).sum())
    outlier_table.append({"column": col, "lower_fence": low, "upper_fence": high,
                          "n_outliers": n_out, "pct": n_out / len(df) * 100})
outlier_df = pd.DataFrame(outlier_table)
print(outlier_df.round(1).to_string(index=False))

# High-priced cars are genuine (BMW, Audi, Mercedes), so expensive cars are KEPT.
# Km_Driven above 500,000 is not believable for a used car of this age (e.g. 806,599 km on a
# 2009 hatchback, 560,000 km on a 2016 model), so those records are treated as data-entry errors.
KM_LIMIT = 500000
bad_km = df[df["Km_Driven"] > KM_LIMIT]
print(f"\nRecords with Km_Driven > {KM_LIMIT:,} (removed as implausible): {len(bad_km)}")
print(bad_km[["Car_Name", "Year", "Km_Driven"]].to_string(index=False))
df = df[df["Km_Driven"] <= KM_LIMIT].reset_index(drop=True)

size_after = df.shape
print("\nDataset size BEFORE cleaning:", size_before)
print("Dataset size AFTER  cleaning:", size_after)
print("Missing values after cleaning:", int(df.isna().sum().sum()))

df_clean = df  # final cleaned dataset used for all analysis below

# =============================================================================
# 4. DESCRIPTIVE STATISTICS
# =============================================================================
print("\n" + "=" * 70)
print("4. DESCRIPTIVE STATISTICS")
print("=" * 70)

num_summary = df_clean[["Year", "Selling_Price", "Km_Driven", "Car_Age"]].describe().T
num_summary["skewness"] = df_clean[["Year", "Selling_Price", "Km_Driven", "Car_Age"]].skew()
print("Numerical columns:")
print(num_summary.round(2))

print("\nCategorical columns:")
print(df_clean[["Brand", "Fuel", "Seller_Type", "Transmission", "Owner"]].describe().T)

for col in ["Fuel", "Seller_Type", "Transmission", "Owner"]:
    print(f"\n{col} share (%):")
    print((df_clean[col].value_counts(normalize=True) * 100).round(1))

print("\nMean vs median price (INR lakh):",
      round(df_clean["Price_Lakh"].mean(), 2), "vs", round(df_clean["Price_Lakh"].median(), 2))

# =============================================================================
# 5. VISUAL EDA  (8 charts, 5 chart types)
# =============================================================================
print("\n" + "=" * 70)
print("5. VISUAL EDA")
print("=" * 70)

# ---- Chart 1: Most common brands (bar chart, univariate) ----
top_brands = df_clean["Brand"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.barplot(x=top_brands.values, y=top_brands.index.astype(str), color=MAIN_COLOR, ax=ax)
for i, v in enumerate(top_brands.values):
    ax.text(v + 8, i, str(v), va="center", fontsize=9)
ax.set_title("Chart 1: Ten most common car brands")
ax.set_xlabel("Number of listings")
ax.set_ylabel("Brand")
save_chart(fig, "chart1_top_brands.png")

# ---- Chart 2: Fuel type distribution (bar chart, univariate) ----
fuel_counts = df_clean["Fuel"].value_counts()
fig, ax = plt.subplots(figsize=(7, 3.8))
sns.barplot(x=fuel_counts.index.astype(str), y=fuel_counts.values, color=MAIN_COLOR, ax=ax)
for i, v in enumerate(fuel_counts.values):
    ax.text(i, v + 25, f"{v} ({v / len(df_clean):.1%})", ha="center", fontsize=9)
ax.set_title("Chart 2: Listings by fuel type")
ax.set_xlabel("Fuel type")
ax.set_ylabel("Number of listings")
save_chart(fig, "chart2_fuel_type.png")

# ---- Chart 3: Price distribution (histogram, univariate) ----
fig, ax = plt.subplots(figsize=(7, 4))
sns.histplot(df_clean["Price_Lakh"], bins=40, log_scale=True, color=MAIN_COLOR, ax=ax)
ax.axvline(df_clean["Price_Lakh"].median(), color=ACCENT_COLOR, linestyle="--",
           label=f"Median = {df_clean['Price_Lakh'].median():.1f} lakh")
ax.axvline(df_clean["Price_Lakh"].mean(), color="black", linestyle=":",
           label=f"Mean = {df_clean['Price_Lakh'].mean():.1f} lakh")
ax.set_title("Chart 3: Distribution of selling price (log scale)")
ax.set_xlabel("Selling price (INR lakh, log scale)")
ax.set_ylabel("Number of cars")
ax.legend()
save_chart(fig, "chart3_price_hist.png")

# ---- Chart 4: Age vs price (scatter, bivariate) ----
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(df_clean["Car_Age"], df_clean["Price_Lakh"], s=12, alpha=0.3, color=MAIN_COLOR)
median_by_age = df_clean.groupby("Car_Age")["Price_Lakh"].median()
ax.plot(median_by_age.index, median_by_age.values, color=ACCENT_COLOR, linewidth=2.2,
        marker="o", markersize=4, label="Median price for each age")
ax.set_yscale("log")
ax.set_title("Chart 4: Car age vs selling price")
ax.set_xlabel("Car age (years)")
ax.set_ylabel("Selling price (INR lakh, log scale)")
ax.legend()
save_chart(fig, "chart4_age_vs_price.png")

# ---- Chart 5: Kilometres driven vs price (scatter, bivariate) ----
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(df_clean["Km_Driven"] / 1000, df_clean["Price_Lakh"], s=12, alpha=0.3,
           color=MAIN_COLOR)
ax.set_yscale("log")
ax.set_title("Chart 5: Kilometres driven vs selling price")
ax.set_xlabel("Kilometres driven (thousand km)")
ax.set_ylabel("Selling price (INR lakh, log scale)")
save_chart(fig, "chart5_km_vs_price.png")

# ---- Chart 6: Price by transmission (box plot, bivariate) ----
fig, ax = plt.subplots(figsize=(6.5, 4.2))
sns.boxplot(data=df_clean, x="Transmission", y="Price_Lakh",
            order=["Manual", "Automatic"], color=MAIN_COLOR, ax=ax)
ax.set_yscale("log")
ax.set_title("Chart 6: Selling price by transmission type")
ax.set_xlabel("Transmission")
ax.set_ylabel("Selling price (INR lakh, log scale)")
save_chart(fig, "chart6_transmission_box.png")

# ---- Chart 7: Average price by brand (bar chart, bivariate) ----
brand_stats = (df_clean.groupby("Brand", observed=True)["Price_Lakh"]
               .agg(["mean", "median", "count"]))
MIN_BRAND_LISTINGS = 20   # ignore brands with too few listings to give a fair average
brand_stats = brand_stats[brand_stats["count"] >= MIN_BRAND_LISTINGS].sort_values(
    "mean", ascending=False)
top_price_brands = brand_stats.head(10)
fig, ax = plt.subplots(figsize=(7, 4.2))
sns.barplot(x=top_price_brands["mean"].values, y=top_price_brands.index.astype(str),
            color=MAIN_COLOR, ax=ax)
for i, (v, n) in enumerate(zip(top_price_brands["mean"], top_price_brands["count"])):
    ax.text(v + 0.3, i, f"{v:.1f}  (n={n})", va="center", fontsize=9)
ax.set_title(f"Chart 7: Average selling price by brand ({MIN_BRAND_LISTINGS}+ listings)")
ax.set_xlabel("Average selling price (INR lakh)")
ax.set_ylabel("Brand")
ax.set_xlim(0, top_price_brands["mean"].max() * 1.2)
save_chart(fig, "chart7_brand_avg_price.png")

# ---- Chart 8: Correlation heatmap (multivariate) ----
corr_data = pd.DataFrame({
    "Price": df_clean["Selling_Price"],
    "Car age": df_clean["Car_Age"],
    "Km driven": df_clean["Km_Driven"],
    "Owner rank": df_clean["Owner"].map(
        {"Test Drive": 0, "First": 1, "Second": 2, "Third": 3, "Fourth+": 4}).astype(int),
    "Automatic (1/0)": (df_clean["Transmission"] == "Automatic").astype(int),
    "Diesel (1/0)": (df_clean["Fuel"] == "Diesel").astype(int),
    "Dealer (1/0)": (df_clean["Seller_Type"] != "Individual").astype(int),
})
corr_matrix = corr_data.corr(method="spearman")
fig, ax = plt.subplots(figsize=(7, 5.4))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1,
            linewidths=0.5, cbar_kws={"label": "Spearman correlation"}, ax=ax)
ax.set_title("Chart 8: Correlation between price and other features")
save_chart(fig, "chart8_correlation_heatmap.png")

print(f"8 charts saved in the '{CHART_DIR}' folder.")

# =============================================================================
# 6. KEY FINDINGS (every number below is computed from the cleaned data)
# =============================================================================
print("\n" + "=" * 70)
print("6. KEY FINDINGS")
print("=" * 70)

findings = {}
n_rows = len(df_clean)

# Finding 1: brand and fuel mix
brand_share = df_clean["Brand"].value_counts(normalize=True) * 100
fuel_share = df_clean["Fuel"].value_counts(normalize=True) * 100
findings["top_brand"] = brand_share.index[0]
findings["top_brand_pct"] = brand_share.iloc[0]
findings["top2_brands_pct"] = brand_share.iloc[:2].sum()
findings["top2_brands"] = list(brand_share.index[:2])
findings["petrol_diesel_pct"] = fuel_share[["Petrol", "Diesel"]].sum()
print(f"1. {findings['top_brand']} is the most common brand ({findings['top_brand_pct']:.1f}%); "
      f"the top two brands make up {findings['top2_brands_pct']:.1f}% of listings. "
      f"Petrol and diesel together are {findings['petrol_diesel_pct']:.1f}% of cars.")

# Finding 2: price is right-skewed
findings["price_mean"] = df_clean["Price_Lakh"].mean()
findings["price_median"] = df_clean["Price_Lakh"].median()
findings["price_max"] = df_clean["Price_Lakh"].max()
findings["price_skew"] = df_clean["Selling_Price"].skew()
findings["pct_under_5_lakh"] = (df_clean["Price_Lakh"] < 5).mean() * 100
print(f"2. Price is right-skewed: mean {findings['price_mean']:.2f} lakh vs median "
      f"{findings['price_median']:.2f} lakh, skewness {findings['price_skew']:.1f}; "
      f"{findings['pct_under_5_lakh']:.1f}% of cars cost under 5 lakh.")

# Finding 3: age is the strongest driver of price
findings["corr_age_price"] = corr_matrix.loc["Price", "Car age"]
slope = np.polyfit(df_clean["Car_Age"], np.log(df_clean["Selling_Price"]), 1)[0]
findings["pct_drop_per_year"] = (1 - np.exp(slope)) * 100
print(f"3. Car age and price have a Spearman correlation of {findings['corr_age_price']:.2f}; "
      f"each extra year of age goes with roughly {findings['pct_drop_per_year']:.0f}% lower price.")

# Finding 4: mileage matters less than age (and is tied to age)
findings["corr_km_price"] = corr_matrix.loc["Price", "Km driven"]
findings["corr_km_age"] = corr_matrix.loc["Km driven", "Car age"]
print(f"4. Km driven has a weaker link with price ({findings['corr_km_price']:.2f}) than age "
      f"does, and km is itself linked with age ({findings['corr_km_age']:.2f}), so part of the "
      f"mileage effect is really an age effect.")

# Finding 5: automatic vs manual
med_trans = df_clean.groupby("Transmission", observed=True)["Price_Lakh"].median()
findings["median_auto"] = med_trans["Automatic"]
findings["median_manual"] = med_trans["Manual"]
findings["auto_ratio"] = med_trans["Automatic"] / med_trans["Manual"]
findings["auto_share"] = (df_clean["Transmission"] == "Automatic").mean() * 100
recent = df_clean[df_clean["Car_Age"] <= 5]
med_recent = recent.groupby("Transmission", observed=True)["Price_Lakh"].median()
findings["auto_ratio_recent"] = med_recent["Automatic"] / med_recent["Manual"]
print(f"5. Automatic cars (only {findings['auto_share']:.1f}% of listings) have a median price of "
      f"{findings['median_auto']:.1f} lakh vs {findings['median_manual']:.1f} lakh for manual "
      f"({findings['auto_ratio']:.1f}x). Among cars up to 5 years old the gap is "
      f"{findings['auto_ratio_recent']:.1f}x.")

# Finding 6: brands with highest average price
findings["brand_price_top3"] = [
    (b, float(r["mean"]), int(r["count"])) for b, r in brand_stats.head(3).iterrows()]
findings["maruti_avg"] = brand_stats.loc["Maruti", "mean"] if "Maruti" in brand_stats.index else np.nan
print(f"6. Highest average price (brands with {MIN_BRAND_LISTINGS}+ listings): " +
      ", ".join(f"{b} {m:.1f} lakh (n={n})" for b, m, n in findings["brand_price_top3"]) +
      f"; Maruti averages {findings['maruti_avg']:.1f} lakh.")

# Finding 7: fuel, seller and owner differences
med_fuel = df_clean.groupby("Fuel", observed=True)["Price_Lakh"].median()
med_seller = df_clean.groupby("Seller_Type", observed=True)["Price_Lakh"].median()
med_owner = df_clean.groupby("Owner", observed=True)["Price_Lakh"].median()
findings["median_diesel"] = med_fuel["Diesel"]
findings["median_petrol"] = med_fuel["Petrol"]
findings["median_dealer"] = med_seller["Dealer"]
findings["median_individual"] = med_seller["Individual"]
findings["median_first_owner"] = med_owner["First"]
findings["median_fourth_owner"] = med_owner["Fourth+"]
print(f"7. Median price: diesel {findings['median_diesel']:.1f} vs petrol "
      f"{findings['median_petrol']:.1f} lakh; dealer {findings['median_dealer']:.1f} vs "
      f"individual {findings['median_individual']:.1f} lakh; first owner "
      f"{findings['median_first_owner']:.1f} vs fourth+ owner "
      f"{findings['median_fourth_owner']:.1f} lakh.")

print("\nLimitations:")
print("- No engine size, horsepower, body type or city columns, so these cannot be analysed.")
print("- Prices are asking prices from one platform, not confirmed sale prices.")
print("- Only one electric car and a few LPG/CNG cars, so those fuels cannot be compared fairly.")
print("- Correlation shows association, not cause (e.g. automatic cars are also newer/pricier brands).")
print("\nEDA complete.")
