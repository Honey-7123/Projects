from pymongo import MongoClient
from datetime import datetime

# Convert forecast rows into JSON-storable format
def convert_forecast_for_mongo(forecast_list):
    return [
        {
            "ds": item["ds"].to_pydatetime() if hasattr(item["ds"], "to_pydatetime") else item["ds"],
            "yhat": float(item["yhat"]),
            "yhat_lower": float(item["yhat_lower"]),
            "yhat_upper": float(item["yhat_upper"])
        }
        for item in forecast_list
    ]

# Save forecast to MongoDB
def save_forecast(store_id, dept_id, forecast):
    # MongoDB connection
    client = MongoClient("mongodb://localhost:27017")
    db = client["forecast_db"]
    collection = db["sales_forecast"]  # ✅ FIXED: not a list

    # Document to insert
    doc = {
        "store_id": int(store_id),
        "dept_id": int(dept_id),
        "forecast": convert_forecast_for_mongo(forecast),  # ✅ FIXED: using correct variable
        "timestamp": datetime.utcnow()
    }

    # Insert document
    collection.insert_one(doc)
