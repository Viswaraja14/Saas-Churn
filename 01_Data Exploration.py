import pandas as pd
df=pd.read_csv("MP_Synthetics_50000_Rows.csv")

# Finding the shape of the dataset
print("Shape of the dataset(ROW,COLUMN):")
print(df.shape)

# basic information about the dataset
print("Basic information about dataset:")
print(df.info())

# Missing values in the dataset
print("Sum of missing values in each column:")
print(df.isnull().sum())

# Duplicates in the dataset
print("Duplicated values:")
print(df.duplicated().sum())

# Target variable distribution
print("Target variable distribution:")
print(df["Is_Churned"].value_counts())

# Unique values in important categorical columns
print("Plan_Tier:")
print(df["Plan_Tier"].unique())

print("Billing_Cycle :")
print(df["Billing_Cycle"].unique())

print("Payment_Status:")
print(df["Payment_Status"].unique())
