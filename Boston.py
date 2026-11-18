import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    df = pd.read_csv("Boston.csv")
    if "MEDV" in df.columns:
        target_column = "MEDV"
    elif "medv" in df.columns:
        target_column = "medv"
    else:
        target_column = df.columns[-1]
except FileNotFoundError:
    from sklearn.datasets import fetch_openml
    dataset = fetch_openml(name="boston", version=1, as_frame=True)
    df = dataset.frame
    target_column = "MEDV" if "MEDV" in df.columns else dataset.target.name

df = df.apply(pd.to_numeric, errors="coerce").dropna()
X = df.drop(columns=[target_column])
y = df[target_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
