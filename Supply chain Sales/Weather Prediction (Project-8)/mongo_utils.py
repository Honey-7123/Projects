from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["air_quality_db"]
collection = db["forecasts"]

def save_forecast(forecast_data):
    document = {"forecast": forecast_data}
    collection.insert_one(document)
