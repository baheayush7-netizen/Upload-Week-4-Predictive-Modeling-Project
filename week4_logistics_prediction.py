"""
Week 4: Predictive Modeling and Optimization in Logistics
Install: pip install pandas numpy scikit-learn matplotlib
Run: python week4_logistics_prediction.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Simulate a reproducible hypothetical dataset
rng = np.random.default_rng(24)
n = 500
distance = rng.integers(30, 1600, n)
volume = rng.integers(5, 600, n)
mode = rng.choice(["Road", "Rail", "Air", "Sea"], n, p=[.55, .22, .15, .08])
region = rng.choice(["North", "South", "East", "West"], n)
traffic = rng.integers(1, 6, n)
weather = rng.choice(["Clear", "Rain", "Storm"], n, p=[.72, .22, .06])
weekend = rng.integers(0, 2, n)
speed = {"Road": 52, "Rail": 68, "Air": 520, "Sea": 32}
base = {"Road": 1.5, "Rail": 2.0, "Air": .5, "Sea": 3.0}
weather_add = {"Clear": 0, "Rain": 1.8, "Storm": 4.5}
region_add = {"North": .3, "South": .6, "East": 1.0, "West": .2}
delivery = []
for i in range(n):
    days = (distance[i] / speed[mode[i]] + base[mode[i]]
            + traffic[i] * .45 + weather_add[weather[i]]
            + region_add[region[i]] + weekend[i] * .5
            + volume[i] / 600 * .7 + rng.normal(0, 1.2))
    delivery.append(max(.5, days))

data = pd.DataFrame({
    "Shipment_ID": [f"W4-{i+1:04d}" for i in range(n)],
    "Distance_km": distance, "Shipment_Volume_kg": volume,
    "Transport_Mode": mode, "Region": region, "Traffic_Level": traffic,
    "Weather": weather, "Weekend_Departure": weekend,
    "Delivery_Time_days": np.round(delivery, 2)
})
data.to_csv("week4_logistics_prediction_dataset.csv", index=False)

# 2. Separate predictors and target; split before fitting to avoid leakage
X = data.drop(columns=["Shipment_ID", "Delivery_Time_days"])
y = data["Delivery_Time_days"]
numeric = ["Distance_km", "Shipment_Volume_kg", "Traffic_Level", "Weekend_Departure"]
categorical = ["Transport_Mode", "Region", "Weather"]
preprocess = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=24
)
models = {
    "Linear Regression": Pipeline([
        ("prep", preprocess), ("model", LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("prep", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=160, max_depth=10, min_samples_leaf=2,
            random_state=24
        ))
    ])
}

for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    rmse = mean_squared_error(y_test, prediction) ** .5
    r2 = r2_score(y_test, prediction)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")

# 3. Five-fold cross-validation on training data for Random Forest
cv_mae = -cross_val_score(
    models["Random Forest"], X_train, y_train,
    cv=5, scoring="neg_mean_absolute_error"
)
print("Random Forest 5-fold CV MAE:", cv_mae.mean(), "+/-", cv_mae.std())

# 4. Refit final model and generate diagnostics
final_model = models["Random Forest"].fit(X_train, y_train)
pred = final_model.predict(X_test)
plt.scatter(y_test, pred, alpha=.6)
plt.xlabel("Actual delivery time (days)")
plt.ylabel("Predicted delivery time (days)")
plt.title("Actual vs Predicted Delivery Time")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)

errors = y_test - pred
plt.figure()
plt.hist(errors, bins=18)
plt.xlabel("Actual - predicted (days)")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.savefig("prediction_errors.png", dpi=150)
plt.show()

# 5. Example operational decision rule (illustrative, not an optimizer):
# flag shipments whose predicted delivery exceeds the service threshold.
service_threshold = 7.0
new_shipments = X_test.copy()
new_shipments["Predicted_Delivery_Days"] = pred
new_shipments["Needs_Review"] = new_shipments["Predicted_Delivery_Days"] > service_threshold
print(new_shipments[["Predicted_Delivery_Days", "Needs_Review"]].head(10))
