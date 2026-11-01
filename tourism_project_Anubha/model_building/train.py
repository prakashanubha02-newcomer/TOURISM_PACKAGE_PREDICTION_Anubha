
# 1. IMPORT NECESSARY LIBRARIES
# ==========================================================

import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


# ==========================================================
# 2. DEFINE PROJECT PATHS
# ==========================================================

DATA_DIR = "tourism_project_Anubha/data"
MODEL_DIR = "tourism_project_Anubha/deployment"

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# 3. CONFIGURE MLFLOW TRACKING
# ==========================================================

mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("tourism-product-prediction")


# ==========================================================
# 4. LOAD TRAINING AND TESTING DATA
# ==========================================================

Xtrain = pd.read_csv(os.path.join(DATA_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(DATA_DIR, "Xtest.csv"))

ytrain = pd.read_csv(
    os.path.join(DATA_DIR, "ytrain.csv")
).squeeze("columns")

ytest = pd.read_csv(
    os.path.join(DATA_DIR, "ytest.csv")
).squeeze("columns")

print("Dataset loaded successfully.")
print("Training shape:", Xtrain.shape)
print("Testing shape:", Xtest.shape)


# ==========================================================
# 5. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ==========================================================

numerical_features = [
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

categorical_features = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation"
]


# ==========================================================
# 6. CALCULATE CLASS WEIGHT TO HANDLE IMBALANCE
# ==========================================================

class_counts = ytrain.value_counts()

class_weight = (
    class_counts.get(0, 0) /
    class_counts.get(1, 1)
)

print("Calculated class weight:", class_weight)


# ==========================================================
# 7. DEFINE DATA PREPROCESSING
# ==========================================================

preprocessor = make_column_transformer(

    (StandardScaler(), numerical_features),

    (OneHotEncoder(handle_unknown="ignore"),
     categorical_features)

)


# ==========================================================
# 8. DEFINE BASE XGBOOST MODEL
# ==========================================================

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss",
    n_jobs=1
)


# ==========================================================
# 9. DEFINE HYPERPARAMETER GRID
# ==========================================================

param_grid = {

    "xgbclassifier__n_estimators": [50, 100],

    "xgbclassifier__max_depth": [2, 3],

    "xgbclassifier__learning_rate": [0.05, 0.1]

}


# ==========================================================
# 10. CREATE MODEL PIPELINE
# ==========================================================

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model
)


# ==========================================================
# 11. START MLFLOW EXPERIMENT
# ==========================================================

with mlflow.start_run():

    # Hyperparameter tuning

    grid_search = GridSearchCV(
        model_pipeline,
        param_grid,
        cv=5,
        scoring="recall",
        n_jobs=-1
    )

    grid_search.fit(Xtrain, ytrain)


    # Log all parameter combinations

    results = grid_search.cv_results_

    for i in range(len(results["params"])):

        param_set = results["params"][i]

        mean_score = results["mean_test_score"][i]

        std_score = results["std_test_score"][i]

        with mlflow.start_run(nested=True):

            mlflow.log_params(param_set)

            mlflow.log_metric("mean_test_score", mean_score)

            mlflow.log_metric("std_test_score", std_score)


    # Log best parameters

    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    print("Best parameters:", grid_search.best_params_)


    # ======================================================
    # 12. MODEL EVALUATION
    # ======================================================

    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]

    y_pred_train = (
        y_pred_train_proba >= classification_threshold
    ).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_test = (
        y_pred_test_proba >= classification_threshold
    ).astype(int)


    train_report = classification_report(
        ytrain,
        y_pred_train,
        output_dict=True,
        zero_division=0
    )

    test_report = classification_report(
        ytest,
        y_pred_test,
        output_dict=True,
        zero_division=0
    )

    print("\nTest Classification Report:")

    print(
        classification_report(
            ytest,
            y_pred_test,
            zero_division=0
        )
    )


    # ======================================================
    # 13. LOG PERFORMANCE METRICS IN MLFLOW
    # ======================================================

    mlflow.log_metrics({

        "train_accuracy": train_report["accuracy"],

        "train_precision": train_report["1"]["precision"],

        "train_recall": train_report["1"]["recall"],

        "train_f1_score": train_report["1"]["f1-score"],

        "test_accuracy": test_report["accuracy"],

        "test_precision": test_report["1"]["precision"],

        "test_recall": test_report["1"]["recall"],

        "test_f1_score": test_report["1"]["f1-score"]

    })

    mlflow.log_param(
        "classification_threshold",
        classification_threshold
    )


    # ======================================================
    # 14. SAVE THE BEST MODEL
    # ======================================================

    model_path = os.path.join(
        MODEL_DIR,
        "best_tourism_product_model.pkl"
    )

    joblib.dump(best_model, model_path)

    # Log model as an MLflow artifact

    mlflow.log_artifact(
        model_path,
        artifact_path="model"
    )

    print("\nModel saved successfully:", model_path)


# ==========================================================
# 15. DISPLAY FINAL RESULTS
# ==========================================================

print("\nExperimentation and Tracking completed successfully!")

print("Training Accuracy:", train_report["accuracy"])

print("Testing Accuracy:", test_report["accuracy"])

print("Testing Precision:", test_report["1"]["precision"])

print("Testing Recall:", test_report["1"]["recall"])

print("Testing F1 Score:", test_report["1"]["f1-score"])
