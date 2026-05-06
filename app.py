# app.py

from models.energy_model import load_and_train_model, forecast_future, suggest_optimization
import numpy as np

def main():
    print("⚡ Energy Forecasting System Starting...\n")

    # Load + train model
    model, features, df = load_and_train_model("data/your_dataset.csv")

    print("\n✅ Model trained successfully!\n")

    # Get last row for prediction
    last_row = df[features].iloc[-1]

    # Forecast next 24 hours
    future_preds = forecast_future(model, last_row, steps=24)

    print("\n🔮 Next 24 Hours Energy Prediction:\n")
    for i, val in enumerate(future_preds):
        print(f"Hour {i}: {round(val, 2)}")

    # Optimization suggestions
    threshold = np.mean(df['energy'])
    suggestions = suggest_optimization(future_preds, threshold)

    print("\n💡 Optimization Suggestions:\n")
    for s in suggestions:
        print(s)


if __name__ == "__main__":
    main()