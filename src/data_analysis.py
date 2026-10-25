import pandas as pd
from pathlib import Path

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 1000)

PARENT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_FILE = PARENT_DIR / "data" / "raw" / "project_risk_raw_dataset.csv"
CLEAN_DATA_FILE = PARENT_DIR / "data" / "processed" / "project_risk_clean_dataset.csv"

df = pd.read_csv(RAW_DATA_FILE)

def init_profile():
  print(f"Rows & Columns: {df.shape}")
  print(df.info())
  print(df.head(1))
  print(df.describe())
  print(df.select_dtypes(include="str").describe())

  print("Negative Values Check:")
  print(df[['Project_Budget_USD', 'Team_Size', 'Estimated_Timeline_Months', 'Complexity_Score']].lt(0).sum())

  print("\nRisk Level Categories:")
  print(df['Risk_Level'].value_counts())

  print("\nProject Type Categories:")
  print(df['Project_Type'].value_counts())

  print("\nDuplicate Project IDs:")
  print(df['Project_ID'].duplicated().sum())
  
def cleaning():
  df_clean = df.copy()

  df_clean = df_clean.drop(columns=['Seasonal_Risk_Factor'])
  df_clean['Project_Start_Month'] = df_clean['Project_Start_Month'].astype(str)

  # For the two columns with ~20% missing, we will fill them with "Unknown"
  df_clean['Change_Control_Maturity'] = df_clean['Change_Control_Maturity'].fillna('Unknown')
  df_clean['Risk_Management_Maturity'] = df_clean['Risk_Management_Maturity'].fillna('Unknown')

  # For the column with 65% missing, let's drop it entirely to avoid bias
  df_clean = df_clean.drop(columns=['Tech_Environment_Stability'])

  print("Cleaning complete! New shape:", df_clean.shape)
  
  return df_clean

def validate_clean_data(df_clean):
  print(f"Total missing values remaining: {df_clean.isnull().sum().sum()}")
  print(df_clean.info())
  print(df_clean['Change_Control_Maturity'].value_counts())
  print('\n')
  print(df_clean['Risk_Management_Maturity'].value_counts())
  print(df_clean[['Project_Start_Month', 'Change_Control_Maturity', 'Risk_Management_Maturity']].info())
  cols_to_check = ['Budget_Utilization_Rate', 'Previous_Delivery_Success_Rate', 'Team_Turnover_Rate']
  print(df_clean[cols_to_check].describe())
  
  # Calculate Q1 (25th percentile) and Q3 (75th percentile)
  Q1 = df_clean['Project_Budget_USD'].quantile(0.25)
  Q3 = df_clean['Project_Budget_USD'].quantile(0.75)
  
  IQR = Q3 - Q1
  
  # Define bounds (anything outside 1.5 * IQR is a statistical outlier)
  lower_bound = Q1 - 1.5 * IQR
  upper_bound = Q3 + 1.5 * IQR
  
  # Count how many outliers exist
  outliers = df_clean[(df_clean['Project_Budget_USD'] < lower_bound) | (df_clean['Project_Budget_USD'] > upper_bound)]
  
  print(f"Number of Budget Outliers: {len(outliers)}")
  print(f"Upper Bound for Budget: ${upper_bound:,.2f}")
  
  # Look at the Project Types and Risk Levels of the 72 budget outliers
  print("Outlier Project Types:")
  print(outliers['Project_Type'].value_counts())
  
  print("\nOutlier Risk Levels:")
  print(outliers['Risk_Level'].value_counts())
  
  # Let's look at the budget of the highest outlier vs the average
  print(f"\nMax Budget of Outliers: ${outliers['Project_Budget_USD'].max():,.2f}")
  
  print("Duplicate Rows:", df_clean.duplicated().sum())
  
  
  df_clean.to_csv(CLEAN_DATA_FILE, index=False)

init_profile()
validate_clean_data(cleaning())