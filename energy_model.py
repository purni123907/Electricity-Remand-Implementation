# models/energy_model.py

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_and_train_model(data_path):
    df = pd.read_csv(data_path)

    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_peak_hour'] = df['hour'].isin([18,19,20,21,22]).astype(int)

    # Lag features
    df['lag_1'] = df['energy'].shift(1)
    df['lag_24'] = df['energy'].shift(24)

    df.dropna(inplace=True)

    features = [
        'temperature', 'hour', 'day_of_week', 'month',
        'is_weekend', 'is_peak_hour', 'lag_1', 'lag_24'
    ]

    X = df[features]
    y = df['energy']

    split = int(len(df) * 0.8)

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6
    )

    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"MAE: {mae}")
    print(f"RMSE: {rmse}")

    return model, features, df


def forecast_future(model, last_row, steps=24):
    predictions = []
    current = last_row.copy()

    for _ in range(steps):
        pred = model.predict(current.values.reshape(1, -1))[0]
        predictions.append(pred)

        # Update lag features
        current['lag_1'] = pred
        current['lag_24'] = current['lag_1']

        # Update time
        current['hour'] = (current['hour'] + 1) % 24

    return predictions


def suggest_optimization(predictions, threshold):
    suggestions = []

    for i, val in enumerate(predictions):
        if val > threshold:
            suggestions.append(f"Hour {i}: High usage → Shift heavy appliances")
        else:
            suggestions.append(f"Hour {i}: Normal usage")

    return suggestions