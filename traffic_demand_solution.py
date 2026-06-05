"""
Traffic Demand Prediction - Solution
=====================================
Model: LightGBM Regressor
Evaluation Metric: max(0, 100 * R² Score)
"""

import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}")

# ─────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineer(df):
    df = df.copy()

    # Parse timestamp "H:M" → hour + minute
    df['hour']        = df['timestamp'].apply(lambda x: int(x.split(':')[0]))
    df['minute']      = df['timestamp'].apply(lambda x: int(x.split(':')[1]))
    df['time_of_day'] = df['hour'] + df['minute'] / 60.0

    # Cyclical encoding of time (captures continuity midnight wrap-around)
    df['hour_sin']  = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['hour'] / 24)
    df['time_sin']  = np.sin(2 * np.pi * df['time_of_day'] / 24)
    df['time_cos']  = np.cos(2 * np.pi * df['time_of_day'] / 24)

    # Encode categorical features
    df['RoadType_enc']      = df['RoadType'].map({'Residential': 0, 'Street': 1, 'Highway': 2}).fillna(-1)
    df['LargeVehicles_enc'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks_enc']     = (df['Landmarks'] == 'Yes').astype(int)
    df['Weather_enc']       = df['Weather'].map({'Sunny': 0, 'Foggy': 1, 'Rainy': 2, 'Snowy': 3}).fillna(-1)

    # Geohash label encoding
    df['geohash_enc'] = pd.Categorical(df['geohash']).codes

    # Impute missing Temperature with median
    df['Temperature'] = df['Temperature'].fillna(df['Temperature'].median())

    # Interaction features
    df['road_lanes']   = df['RoadType_enc'] * df['NumberofLanes']
    df['is_rush_hour'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    return df

train_fe = feature_engineer(train)
test_fe  = feature_engineer(test)

FEATURES = [
    'day', 'hour', 'minute', 'time_of_day',
    'hour_sin', 'hour_cos', 'time_sin', 'time_cos',
    'RoadType_enc', 'NumberofLanes', 'LargeVehicles_enc',
    'Landmarks_enc', 'Temperature', 'Weather_enc',
    'geohash_enc', 'road_lanes', 'is_rush_hour', 'is_night'
]

X      = train_fe[FEATURES]
y      = train_fe['demand']
X_test = test_fe[FEATURES]

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
model = LGBMRegressor(
    n_estimators=1500,
    learning_rate=0.03,
    num_leaves=128,
    max_depth=8,
    min_child_samples=20,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2', n_jobs=-1)
print(f"\n5-Fold CV R² scores: {cv_scores.round(4)}")
print(f"Mean CV R²:  {cv_scores.mean():.4f}")
print(f"Scaled score (0–100): {max(0, 100 * cv_scores.mean()):.2f}")

# Train on full training data
model.fit(X, y)

# ─────────────────────────────────────────────
# 4. PREDICTIONS & SUBMISSION
# ─────────────────────────────────────────────
preds = model.predict(X_test)
preds = np.clip(preds, 0, 1)          # demand is bounded [0, 1]

submission = pd.DataFrame({'Index': test['Index'], 'demand': preds})
submission.to_csv('submission.csv', index=False)

print(f"\nSubmission file saved — shape: {submission.shape}")
print(submission.head())
print("\nPrediction stats:")
print(submission['demand'].describe())
