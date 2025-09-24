import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. read data ---
train = pd.read_csv("train.csv")

# --- 2. date features ---
def add_date_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df = df.drop(columns=["date"])
    return df

train = add_date_features(train)

# --- 3. data cleaning ---
X = train.drop(columns=["id", "demand"], errors="ignore")
y = train["demand"]

# --- 4. train split & validation ---
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# --- 5. train prediction ---
y_pred = model.predict(X_val)

# --- 7. model eval ---
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)
print(f"[Validation] RMSE: {rmse:.2f}")
print(f"[Validation] R²: {r2:.3f}")

# --- 8. plot ---
plt.figure(figsize=(7,6))
plt.scatter(y_val, y_pred, alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
         color="red", linestyle="--")
plt.xlabel("Actual demand")
plt.ylabel("Predicted demand")
plt.title("LightGBM with Date Features - Validation Set")
plt.grid(True)
plt.show()

# --- 9. read test.csv & date featuring ---
test = pd.read_csv("test.csv")
test = add_date_features(test)
X_test = test.drop(columns=["id"], errors="ignore")

# --- 10. train all set ---
final_model = lgb.LGBMRegressor(random_state=42)
final_model.fit(X, y)

# --- 11. predict test.csv ---
y_test_pred = final_model.predict(X_test)

# --- 12. save results ---
submission = pd.DataFrame({
    "id": test["id"],
    "demand": y_test_pred
})
submission.to_csv("prediction.csv", index=False)

print("prediction.csv has been produced.")
