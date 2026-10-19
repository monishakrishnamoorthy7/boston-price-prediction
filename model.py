from pathlib import Path
import pickle

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "boston.csv")
X = data.drop("medv", axis=1)
y = data["medv"]
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

with (BASE_DIR / "model.pkl").open("wb") as model_file:
    pickle.dump(model, model_file)
print("Model saved as model.pkl")
