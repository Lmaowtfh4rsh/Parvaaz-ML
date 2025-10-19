import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import time

start_time = time.time()

try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
except FileNotFoundError:
    print("Error: Make sure 'train.csv' and 'test.csv' are in the correct folder.")
    exit()

test_ids = df_test['Id']

df_train_id_dropped = df_train.drop('Id', axis=1)
df_test_id_dropped = df_test.drop('Id', axis=1)

for df in [df_train_id_dropped, df_test_id_dropped]:
    df['Lifestyle Activities'] = df['Lifestyle Activities'].apply(lambda x: 1 if x == 'Yes' else 0)

features = ['Therapy Hours', 'Initial Health Score', 'Lifestyle Activities', 'Average Sleep Hours', 'Follow-Up Sessions']
target = 'Recovery Index'

X = df_train_id_dropped[features]
y = df_train_id_dropped[target]
X_final_test = df_test_id_dropped[features]

regressor = DecisionTreeRegressor(random_state=42)
rfe_selector = RFE(estimator=regressor, n_features_to_select=3, step=1)

pipeline = Pipeline([
    ('feature_selection', rfe_selector),
    ('regression', regressor)
])

kfold = KFold(n_splits=10, shuffle=True, random_state=42)

scores = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_root_mean_squared_error')

mean_rmse = -np.mean(scores)
std_rmse = np.std(scores)

cv_end_time = time.time()

pipeline.fit(X, y)

final_predictions = pipeline.predict(X_final_test)

final_predictions_clipped = np.clip(final_predictions, 10, 100)
final_predictions_rounded = np.round(final_predictions_clipped).astype(int)

submission_df = pd.DataFrame({'Id': test_ids, 'Recovery Index': final_predictions_rounded})
submission_df.to_csv('recovery_predictions_kfold.csv', index=False)

final_end_time = time.time()

final_selector = pipeline.named_steps['feature_selection']
selected_features = X.columns[final_selector.support_].tolist()

print("--- Model Training & K-Fold Evaluation Complete ---")
print(f"Top 3 Features Selected by RFE: {selected_features}")
print(f"Cross-Validation RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
print(f"Cross-validation took: {cv_end_time - start_time:.2f} seconds")
print(f"Total time (including final fit/predict): {final_end_time - start_time:.2f} seconds")
print("\nPredictions for the test dataset have been successfully generated.")
print("The results are saved in the file: 'recovery_predictions_kfold.csv'")
print("\nSample Submission:")
print(submission_df.head())