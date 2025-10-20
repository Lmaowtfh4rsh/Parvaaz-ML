# ==========================
# Model 4: Bayesian Ridge Regression
# ==========================

# --- Imports ---
import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error

# --- Setup ---
RSEED = 42
np.random.seed(RSEED)

# --- Load Data ---
train = pd.read_csv("data/train.csv")

# --- Preprocessing ---
# Encode Lifestyle Activities (Yes/No → 1/0)
train["Lifestyle Activities"] = train["Lifestyle Activities"].map({"Yes": 1, "No": 0})

# Handle missing values (if any)
train = train.fillna(train.mean())

# Separate features and target
X = train.drop(columns=["Recovery Index"])
y = train["Recovery Index"]

# --- Define Pipeline ---
bayesian_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', BayesianRidge())
])

# --- Define RMSE Scorer ---
rmse_scorer = make_scorer(mean_squared_error, squared=False, greater_is_better=False)

# --- 10-Fold Cross-Validation ---
scores = cross_val_score(
    bayesian_pipe, X, y,
    cv=10,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

mean_rmse = -np.mean(scores)
std_rmse = np.std(scores)

print("========================================")
print("Bayesian Ridge Regression Results")
print(f"Mean RMSE (10-fold CV): {mean_rmse:.4f}")
print(f"Std Dev of RMSE: {std_rmse:.4f}")
print("========================================")

# --- Train Final Model on Full Data ---
bayesian_pipe.fit(X, y)

# --- Load Test Data ---
test = pd.read_csv("data/test.csv")

# Apply same preprocessing to test set
test["Lifestyle Activities"] = test["Lifestyle Activities"].map({"Yes": 1, "No": 0})
test = test.fillna(train.mean())

# --- Predict on Test Set ---
preds = bayesian_pipe.predict(test[X.columns])

# Clip and round predictions as per problem statement
preds = np.round(np.clip(preds, 10, 100)).astype(int)

# --- Save Submission File ---
submission = pd.DataFrame({
    "Id": test["Id"],
    "Recovery Index": preds
})

submission.to_csv("submission_bayesianridge.csv", index=False)

print("✅ Submission file 'submission_bayesianridge.csv' successfully created!")