import pandas as pd
import glob
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import pickle

# --- Step 1: Load all CSV files ---
csv_files = glob.glob("*.csv")
combined = pd.DataFrame()

print("🔍 Found CSV files:", csv_files)

for file in csv_files:
    try:
        df = pd.read_csv(file)
        print(f"\n📄 {file} — Shape: {df.shape}")
        print(f"📌 Columns: {df.columns.tolist()}")
        df['city'] = os.path.splitext(os.path.basename(file))[0].capitalize()
        combined = pd.concat([combined, df], ignore_index=True)
    except Exception as e:
        print(f"❌ Error loading {file}: {e}")

# --- Step 2: Rename relevant columns ---
rename_map = {
    "Area": "area",
    "Size": "bhk",
    "No. of Bedrooms": "bhk",
    "bedroom": "bhk",
    "BHK": "bhk",
    "Price": "price"
}
combined.rename(columns=rename_map, inplace=True)

# --- Step 3: Drop duplicate columns if any ---
combined = combined.loc[:, ~combined.columns.duplicated()]

# --- Step 4: Check if required columns exist ---
required_columns = ["city", "area", "bhk", "price"]
available = [col for col in required_columns if col in combined.columns]
missing = [col for col in required_columns if col not in available]

print("\n✅ Columns available for training:", available)
if missing:
    print("⚠️ Missing columns:", missing)

if len(available) < 4:
    raise ValueError("🚫 Not enough usable columns. Fix column names in CSV files.")

combined = combined[required_columns]

# --- Step 5: Clean area, bhk, price values ---
def extract_number(val):
    if isinstance(val, str):
        match = re.search(r"\d+\.?\d*", val.replace(',', ''))
        return float(match.group()) if match else None
    return val

for col in ["area", "bhk", "price"]:
    print(f"\n🔍 Cleaning column: {col}")
    combined[f"{col}_raw"] = combined[col].astype(str)
    combined[col] = combined[col].apply(extract_number)

print("\n🧾 Area extraction preview:\n", combined[["area_raw", "area"]].head())
print("\n🧾 BHK extraction preview:\n", combined[["bhk_raw", "bhk"]].head())
print("\n🧾 Price extraction preview:\n", combined[["price_raw", "price"]].head())

# --- Step 6: Filter data ---
print("\n🔍 Before filtering → Rows:", len(combined))
combined.dropna(subset=["area", "bhk", "price"], inplace=True)
combined = combined[(combined["area"] > 100) & (combined["bhk"] <= 10)]
combined = combined[combined["price"] < 10**8]

# --- Step 7: Save cleaned data ---
combined = combined[["city", "area", "bhk", "price"]]
combined.to_csv("real_estate_data.csv", index=False)
print(f"\n📦 Combined dataset saved as real_estate_data.csv → Rows: {combined.shape[0]}")

# --- Step 8: Train model ---
X = combined[["city", "area", "bhk"]]
y = combined["price"]

cat_cols = ["city"]
num_cols = ["area", "bhk"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(n_estimators=100, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# --- Step 9: Save model ---
with open("xgboost.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("✅ Model trained and saved as xgboost.pkl")
