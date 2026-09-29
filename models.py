import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from xgboost import XGBRegressor

def qlike(actual, predicted):
    epsilon = 1e-8

    actual_var = np.maximum(np.array(actual) ** 2, epsilon)
    predicted_var = np.maximum(np.array(predicted) ** 2, epsilon)

    return np.mean(
        np.log(predicted_var) +
        actual_var / predicted_var
    )


data = pd.read_csv(
    "FinPulse_Processed_Data.csv",
    parse_dates=["Date"]
)

print(data.shape)

n = len(data)

train_end = int(n * 0.60)
val_end = int(n * 0.80)

train = data.iloc[:train_end].copy()
validation = data.iloc[train_end:val_end].copy()
test = data.iloc[val_end:].copy()

print("\nTrain shape:", train.shape)
print("Validation shape:", validation.shape)
print("Test shape:", test.shape)

print("\nTrain period:")
print(train["Date"].min(), "to", train["Date"].max())

print("\nValidation period:")
print(validation["Date"].min(), "to", validation["Date"].max())

print("\nTest period:")
print(test["Date"].min(), "to", test["Date"].max())

test["Naive_Prediction"] = test["Abs_Return"]

test["Naive_Prediction"] = test["Abs_Return"]

print(
    test[
        ["Date", "Abs_Return", "Target", "Naive_Prediction"]
    ].head(10)
)


mae = mean_absolute_error(
    test["Target"],
    test["Naive_Prediction"]
)

rmse = np.sqrt(
    mean_squared_error(
        test["Target"],
        test["Naive_Prediction"]
    )
)

print("\nNaive Model Results")
print("MAE:", mae)
print("RMSE:", rmse)

#.................
#GARCH MODEL
#.................

from arch import arch_model

garch_predictions = []

test_start = val_end

for i in range(test_start, len(data)):

    history = data["Return"].iloc[:i + 1]

    model = arch_model(
        history,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    fitted_model = model.fit(disp="off")

    forecast = fitted_model.forecast(horizon=1)

    next_variance = forecast.variance.iloc[-1, 0]

    next_volatility = np.sqrt(next_variance)

    garch_predictions.append(next_volatility)

print("Number of GARCH predictions:", len(garch_predictions))
print("Number of test rows:", len(test))

test["GARCH_Prediction"] = garch_predictions

garch_mae = mean_absolute_error(
    test["Target"],
    test["GARCH_Prediction"]
)

garch_rmse = np.sqrt(
    mean_squared_error(
        test["Target"],
        test["GARCH_Prediction"]
    )
)

print("\nGARCH(1,1) Results")
print("MAE:", garch_mae)
print("RMSE:", garch_rmse)

print("\nModel Comparison")

print("Naive")
print("MAE :", mae)
print("RMSE:", rmse)

print("\nGARCH")
print("MAE :", garch_mae)
print("RMSE:", garch_rmse)

# -----------------------------
# TGARCH / TARCH MODEL
# -----------------------------

tgarch_predictions = []

test_start = val_end

for i in range(test_start, len(data)):

    # Use all returns available up to the current day
    history = data["Return"].iloc[:i + 1]

    # Fit TGARCH / TARCH(1,1)
    model = arch_model(
        history,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        power=1.0,
        dist="normal"
    )

    fitted_model = model.fit(disp="off")

    # Forecast next day
    forecast = fitted_model.forecast(horizon=1)

    next_variance = forecast.variance.iloc[-1, 0]
    next_volatility = np.sqrt(next_variance)

    tgarch_predictions.append(next_volatility)

print("Number of TGARCH predictions:", len(tgarch_predictions))
print("Number of test rows:", len(test))

test["TGARCH_Prediction"] = tgarch_predictions

p=1,
q=1
o=1
power=1.0

tgarch_mae = mean_absolute_error(
    test["Target"],
    test["TGARCH_Prediction"]
)

tgarch_rmse = np.sqrt(
    mean_squared_error(
        test["Target"],
        test["TGARCH_Prediction"]
    )
)

print("\nTGARCH Results")
print("MAE:", tgarch_mae)
print("RMSE:", tgarch_rmse)

# -----------------------------
# EGARCH MODEL
# -----------------------------

egarch_predictions = []

test_start = val_end

for i in range(test_start, len(data)):

    # Returns available up to the current day
    history = data["Return"].iloc[:i + 1]

    # Fit EGARCH(1,1)
    model = arch_model(
        history,
        mean="Constant",
        vol="EGARCH",
        p=1,
        o=1,
        q=1,
        dist="normal"
    )

    fitted_model = model.fit(disp="off")

    # One-step-ahead forecast
    forecast = fitted_model.forecast(horizon=1)

    next_variance = forecast.variance.iloc[-1, 0]
    next_volatility = np.sqrt(next_variance)

    egarch_predictions.append(next_volatility)

print("Number of EGARCH predictions:", len(egarch_predictions))
print("Number of test rows:", len(test))

test["EGARCH_Prediction"] = egarch_predictions

egarch_mae = mean_absolute_error(
    test["Target"],
    test["EGARCH_Prediction"]
)

egarch_rmse = np.sqrt(
    mean_squared_error(
        test["Target"],
        test["EGARCH_Prediction"]
    )
)

print("\nEGARCH Results")
print("MAE:", egarch_mae)
print("RMSE:", egarch_rmse)

feature_cols = [
    "Return",
    "Abs_Return",
    "Vol_5",
    "Vol_10",
    "Vol_21",
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "AbsReturn_Lag1",
    "AbsReturn_Lag2",
    "AbsReturn_Lag5",
    "VIX",
    "VIX_Lag1",
    "VIX_Change_1D",
    "VIX_Change_5D",
    "Parkinson",
    "Volume_Change_1D"
]

X_train = train[feature_cols]
y_train = train["Target"]

X_val = validation[feature_cols]
y_val = validation["Target"]

X_test = test[feature_cols]
y_test = test["Target"]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

alphas = [0.01, 0.1, 1, 10, 100]

best_alpha = None
best_val_mae = float("inf")

for alpha in alphas:

    ridge = Ridge(alpha=alpha)

    ridge.fit(
        X_train_scaled,
        y_train
    )

    val_predictions = ridge.predict(
        X_val_scaled
    )

    val_mae = mean_absolute_error(
        y_val,
        val_predictions
    )

    print(
        "Alpha:",
        alpha,
        "Validation MAE:",
        val_mae
    )

    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_alpha = alpha

print("\nBest Ridge Alpha:", best_alpha)

train_val = pd.concat(
    [train, validation]
)

X_train_val = train_val[feature_cols]
y_train_val = train_val["Target"]

final_scaler = StandardScaler()

X_train_val_scaled = final_scaler.fit_transform(
    X_train_val
)

X_test_scaled = final_scaler.transform(
    X_test
)

ridge_model = Ridge(
    alpha=best_alpha
)

ridge_model.fit(
    X_train_val_scaled,
    y_train_val
)

ridge_predictions = ridge_model.predict(
    X_test_scaled
)

ridge_predictions = np.clip(
    ridge_predictions,
    0,
    None
)

test["Ridge_Prediction"] = ridge_predictions

ridge_mae = mean_absolute_error(
    y_test,
    ridge_predictions
)

ridge_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        ridge_predictions
    )
)

print("\nRidge Results")
print("MAE :", ridge_mae)
print("RMSE:", ridge_rmse)

best_params = None
best_val_mae = float("inf")

learning_rates = [0.01, 0.05, 0.1]
max_depths = [2, 3, 4]

for lr in learning_rates:
    for depth in max_depths:

        model = XGBRegressor(
            n_estimators=300,
            learning_rate=lr,
            max_depth=depth,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42
        )

        model.fit(
            X_train,
            y_train
        )

        val_predictions = model.predict(
            X_val
        )

        val_predictions = np.clip(
            val_predictions,
            0,
            None
        )

        val_mae = mean_absolute_error(
            y_val,
            val_predictions
        )

        print(
            "Learning rate:",
            lr,
            "| Max depth:",
            depth,
            "| Validation MAE:",
            val_mae
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = {
                "learning_rate": lr,
                "max_depth": depth
            }

print("\nBest XGBoost parameters:")
print(best_params)

print("Best validation MAE:")
print(best_val_mae)

train_val = pd.concat(
    [train, validation]
)

X_train_val = train_val[feature_cols]
y_train_val = train_val["Target"]

xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=best_params["learning_rate"],
    max_depth=best_params["max_depth"],
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)

xgb_model.fit(
    X_train_val,
    y_train_val
)

xgb_predictions = xgb_model.predict(
    X_test
)

xgb_predictions = np.clip(
    xgb_predictions,
    0,
    None
)

test["XGBoost_Prediction"] = xgb_predictions

xgb_mae = mean_absolute_error(
    y_test,
    xgb_predictions
)

xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_predictions
    )
)

print("\nXGBoost Results")
print("MAE :", xgb_mae)
print("RMSE:", xgb_rmse)

# Naive
naive_qlike = qlike(
    test["Target"],
    test["Naive_Prediction"]
)

# GARCH
garch_qlike = qlike(
    test["Target"],
    test["GARCH_Prediction"]
)

# TGARCH
tgarch_qlike = qlike(
    test["Target"],
    test["TGARCH_Prediction"]
)

# EGARCH
egarch_qlike = qlike(
    test["Target"],
    test["EGARCH_Prediction"]
)

# Ridge
ridge_qlike = qlike(
    test["Target"],
    test["Ridge_Prediction"]
)

# XGBoost
xgb_qlike = qlike(
    test["Target"],
    test["XGBoost_Prediction"]
)

results = pd.DataFrame({
    "Model": [
        "Naive",
        "GARCH",
        "TGARCH",
        "EGARCH",
        "Ridge",
        "XGBoost"
    ],

    "MAE": [
        mae,
        garch_mae,
        tgarch_mae,
        egarch_mae,
        ridge_mae,
        xgb_mae
    ],

    "RMSE": [
        rmse,
        garch_rmse,
        tgarch_rmse,
        egarch_rmse,
        ridge_rmse,
        xgb_rmse
    ],

    "QLIKE": [
        naive_qlike,
        garch_qlike,
        tgarch_qlike,
        egarch_qlike,
        ridge_qlike,
        xgb_qlike
    ]
})

print("\nFINAL MODEL COMPARISON")
print(results.round(4))

joblib.dump(ridge_model, "ridge_model.pkl")
joblib.dump(final_scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")

print("Ridge model and scaler saved successfully.")





#py models.py