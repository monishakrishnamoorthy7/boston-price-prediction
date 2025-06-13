import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

# Load dataset from CSV
data = pd.read_csv("boston.csv")

# Split features and target
X = data.drop("medv", axis=1)
y = data["medv"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("✅ Model saved as model.pkl")
