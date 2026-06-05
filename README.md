# Traffic Demand Prediction - HackerEarth

Solution for the Traffic Demand Prediction hackathon challenge.

## Problem Statement

Predict traffic demand at different geo-locations based on road type, weather, time, and other features.

**Evaluation:** `max(0, 100 * R2_score(actual, predicted))`

## Approach

### Feature Engineering
- Parsed timestamp (H:M) into hour and minute
- Added cyclical features (sin/cos) for time of day
- Encoded categorical features: RoadType, Weather, LargeVehicles, Landmarks
- Label encoded geohash (1,249 unique locations)
- Filled missing Temperature with median
- Created interaction features: road_lanes, is_rush_hour, is_night

### Model
- **LightGBM Regressor**
- 1500 estimators, learning_rate=0.03
- 5-Fold Cross Validation for evaluation
- Predictions clipped to [0, 1]

## Files
- `traffic_demand_solution.py` - Main Python solution
- `traffic_demand_prediction.ipynb` - Detailed notebook with EDA
- `submission.csv` - Final predictions

## How to Run
```bash
pip install pandas numpy lightgbm scikit-learn
python traffic_demand_solution.py
```
