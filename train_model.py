import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("loan_data.csv")

# Manual encoding
df["Gender"] = df["Gender"].map({
    "Male": 1,
    "Female": 0
})

df["Married"] = df["Married"].map({
    "Yes": 1,
    "No": 0
})

df["Education"] = df["Education"].map({
    "Graduate": 1,
    "Not Graduate": 0
})

df["Self_Employed"] = df["Self_Employed"].map({
    "Yes": 1,
    "No": 0
})

df["Property_Area"] = df["Property_Area"].map({
    "Rural": 0,
    "Semiurban": 1,
    "Urban": 2
})

df["Loan_Status"] = df["Loan_Status"].map({
    "N": 0,
    "Y": 1
})

# Features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model
joblib.dump(model, "loan_model.pkl")

print("Model saved as loan_model.pkl")