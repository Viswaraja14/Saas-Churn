from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#load data
df=pd.read_csv("cleaned_churn_data.csv")

# SEPRATE FEATURES AND TARGET VARIABLE
X = df.drop(columns=["Is_Churned"], axis=1)
y = df["Is_Churned"]

# Convert categorical variables into numerical variables
X = pd.get_dummies(
    X,
    drop_first=True
)

# Target variable is already encoded as 0 and 1
y = df["Is_Churned"].astype(int)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(
    max_iter=1000
)

# Drop columns that contain no usable values (for example, the two all-NaN columns)
all_missing_cols = X_train.columns[X_train.isna().all()]
X_train = X_train.drop(columns=all_missing_cols)
X_test = X_test.drop(columns=all_missing_cols)

# Use training medians for both sets
medians = X_train.median()
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)

model.fit(X_train, y_train)


print("Model training completed.")

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Number of features:", X_train.shape[1])
