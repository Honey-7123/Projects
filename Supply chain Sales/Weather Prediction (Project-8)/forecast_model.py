from prophet import Prophet
from prepare_data import load_and_preprocess_data

def train_prophet(periods=7):
    df = load_and_preprocess_data()
    df = df.rename(columns={"DateTime": "ds", "CO_concentration": "y"})

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
