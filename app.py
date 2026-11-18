import joblib
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="Iris Classification API",
    version="1.0.0"
)

MODEL_PATH = Path(__file__).parent / "model.pkl"

# Load the trained model.
# If it does not exist, train a default model.
try:
    model = joblib.load(MODEL_PATH)

except FileNotFoundError:
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    data = load_iris()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(data.data, data.target)

    joblib.dump(model, MODEL_PATH)


class IrisInput(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(input_data: IrisInput):

    if len(input_data.features) != 4:
        raise HTTPException(
            status_code=400,
            detail="Exactly 4 features required."
        )

    prediction = model.predict(
        [input_data.features]
    )

    return {
        "model_version": "1.0.0",
        "prediction": int(prediction[0])
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "version": "1.0.0"
    }