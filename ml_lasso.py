import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import time

start_time = time.time()

try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: Make sure 'train.csv' and 'test.csv' are in the correct folder.")
    exit()

test_ids = test_df['Id']

train_df_no_id = train_df.drop('Id', axis=1)
test_df_no_id = test_df.drop('Id', axis=1)

for df in [train_df_no_id, test_df_no_id]:
    df['Lifestyle Activities'] = df['Lifestyle Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

features = ['Therapy Hours', 'Initial Health Score', 'Lifestyle Activities', 'Average Sleep Hours', 'Follow-Up Sessions']
target = 'Recovery Index'

X_train = train_df_no_id[features]
y_train = train_df_no_id[target]
X_test = test_df_no_id[features]

model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', Lasso(alpha=1.0, random_state=42))
])

kfold_cv = KFold(n_splits=10, shuffle=True, random_state=42)

cv_scores = cross_val_score(model_pipeline, X_train, y_train, cv=kfold_cv, scoring='neg_root_mean_squared_error')

mean_rmse = -np.mean(cv_scores)
std_rmse = np.std(cv_scores)

cv_end_time = time.time()

model_pipeline.fit(X_train, y_train)

test_predictions = model_pipeline.predict(X_test)

clipped_predictions = np.clip(test_predictions, 10, 100)
rounded_predictions = np.round(clipped_predictions).astype(int)

submission_df = pd.DataFrame({'Id': test_ids, 'Recovery Index': rounded_predictions})
submission_df.to_csv('recovery_predictions_lasso_kfold.csv', index=False)

final_end_time = time.time()

print("--- Lasso Regression Model Training & K-Fold Evaluation Complete ---")
print(f"Cross-Validation RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
print(f"Cross-validation took: {cv_end_time - start_time:.2f} seconds")
print(f"Total time (including final fit/predict): {final_end_time - start_time:.2f} seconds")
print("\nPredictions for the test dataset have been successfully generated.")
print("The results are saved in the file: 'recovery_predictions_lasso_kfold.csv'")
print("\nSample Submission:")
print(submission_df.head())