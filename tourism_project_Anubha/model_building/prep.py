# for data manipulation
import os
import pandas as pd
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 1. Define file paths
DATA_DIR = "tourism_project_Anubha/data"

RAW_PATH = os.path.join(DATA_DIR, "tourism.csv")

# 2. Load the registered dataset
df = pd.read_csv(RAW_PATH)

print("Dataset loaded successfully.")
print("Original shape:", df.shape)

# 3. Drop serial number and customer ID
df.drop(
    columns=["Unnamed: 0", "CustomerID"],
    errors="ignore",
    inplace=True
)

# 4. Clean the Gender column
df["Gender"] = (
    df["Gender"]
    .astype("string")
    .str.strip() # CLEAN OUTER SPACES
    .str.replace(r"\s+", " ", regex=True) # CLEANS REPEATED SPACES
    .str.title() # STANDARDISES CAPITALIZATION
)

df["Gender"] = df["Gender"].replace({
    "Fe Male": "Female",
    "FeMale" : "Female",
    "Female": "Female",
    "Male": "Male"
})

# 5. Standardize MaritalStatus
df["MaritalStatus"] = (
    df["MaritalStatus"]
    .astype("string")
    .str.strip() # CLEAN OUTER SPACES
    .str.replace(r"\s+", " ", regex=True) # CLEANS REPEATED SPACES
    .str.title() # STANDARDISES CAPITALIZATION
)

df["MaritalStatus"] = df["MaritalStatus"].replace({
    "Unmarried": "Single"
})

# 6. Define target variable
target = "ProdTaken"

# 7. Define numerical features
numeric_features = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome"
]

# 8. Define categorical features
categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]

# 9. Validate expected columns
expected_columns = (
    numeric_features + categorical_features + [target]
)

missing_columns = [
    col for col in expected_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

# 10. Separate predictors (X) and target (y)
X = df[numeric_features + categorical_features]
y = df[target]

# 11. Split into training and testing datasets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# 12. Save the four files for the next MLOps stage LOCALLY
Xtrain.to_csv(
    os.path.join(DATA_DIR, "Xtrain.csv"), index=False
)
Xtest.to_csv(
    os.path.join(DATA_DIR, "Xtest.csv"), index=False
)
ytrain.to_frame().to_csv(
    os.path.join(DATA_DIR, "ytrain.csv"), index=False
)
ytest.to_frame().to_csv(
    os.path.join(DATA_DIR, "ytest.csv"), index=False
)

# 13. Display results
print("\nData preparation completed successfully!")
print("Features:", X.shape[1])
print("Training rows:", len(Xtrain))
print("Testing rows:", len(Xtest))

print("\nTraining target distribution:")
print(ytrain.value_counts())

print("\nTesting target distribution:")
print(ytest.value_counts())

print("\nCleaned categories:")
print("Gender:", df["Gender"].value_counts().to_dict())
print(
    "Marital Status:",
    df["MaritalStatus"].value_counts().to_dict()
)

print("\nTrain/test files saved in:", DATA_DIR)

# 14. Identify categorical and numerical columns
categorical_features = Xtrain.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = Xtrain.select_dtypes(
    include=["number"]
).columns.tolist()

# 15. Create One-Hot Encoder
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ],
    remainder="passthrough"
)

# Fit encoder on training data
Xtrain_encoded = preprocessor.fit_transform(Xtrain) #Learns the categories from training data and converts them into numerical columns.

# Apply same encoder to testing data
Xtest_encoded = preprocessor.transform(Xtest) #Applies the already learned encoding to testing data without learning anything new to prevent data leakage.
