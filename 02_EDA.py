# Load data
df=pd.read_csv("MP_Synthetics_50000_Rows.csv")

#REMOVING UNNECESSARY COLUMNS
df = df.drop(columns=["Customer_ID", "Company_Name"])

#REMOVING DUPLICATES
df=df.drop_duplicates()
print(df.info())

#delet the duplicate columns
df = df.loc[:, ~df.columns.duplicated()]

#cleaning txt columns
categorical_cols = [
    "Industry",
    "Country",
    "Company_Size",
    "Account_Manager",
    "Plan_Tier",
    "Billing_Cycle",
    "Payment_Status"
]
print(categorical_cols)

for col in categorical_cols:
    df[col] = df[col].str.strip().str.title()

df["Signup_Date"] = pd.to_datetime(df["Signup_Date"])
df["Last_Active_Date"] = pd.to_datetime(df["Last_Active_Date"])

df["Feature_Adoption_Rate"] = pd.to_numeric(
    df["Feature_Adoption_Rate"], errors="coerce"
)

df["Churn_Risk_Score"] = pd.to_numeric(
    df["Churn_Risk_Score"], errors="coerce"
)

# Display final results
print("Cleaned DataFrame Shape:")
print(df.shape)

print("Remaining Missing Values:")
print(df.isnull().sum())
# Save cleaned dataset
df.to_csv("cleaned_churn_data.csv", index=False)

print("Cleaned dataset saved successfully..")
