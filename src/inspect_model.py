
import joblib
import pandas as pd

MODEL_PATH = "artifacts/ranker_logistic_regression.joblib"

# Load trained model
artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
features = artifact["features"]

# Extract Logistic Regression coefficients
classifier = model.named_steps["classifier"]

importance = pd.DataFrame({
    "feature": features,
    "coefficient": classifier.coef_[0]
})

# Sort by absolute coefficient
importance["absolute_coefficient"] = (
    importance["coefficient"].abs()
)

importance = importance.sort_values(
    "absolute_coefficient",
    ascending=False
)

print("\n========== MODEL FEATURE IMPORTANCE ==========")

print(
    importance[
        ["feature", "coefficient"]
    ].to_string(index=False)
)

print("\n========== INTERPRETATION ==========")

print("Positive coefficient:")
print("Higher feature values increase the predicted positive score.")

print("\nNegative coefficient:")
print("Higher feature values decrease the predicted positive score.")
