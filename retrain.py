
import joblib
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save the trained model in the project root
model_path = Path(__file__).parent / "model.pkl"
joblib.dump(model, model_path)

print(f"Model trained and saved successfully: {model_path}")
