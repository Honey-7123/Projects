from forecast_model import train_prophet

if __name__ == "__main__":
    forecast = train_prophet()
    print(forecast)
