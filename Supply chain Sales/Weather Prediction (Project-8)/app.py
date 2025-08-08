from fastapi import FastAPI
from forecast_model import train_prophet
from mongo_utils import save_forecast

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Air Quality Forecast API running"}

@app.get("/forecast")
def get_forecast():
    try:
        forecast = train_prophet()
        forecast_data = forecast.to_dict(orient="records")

        save_forecast(forecast_data)

        return {"forecast": forecast_data}
    except Exception as e:
        return {"error": str(e)}
