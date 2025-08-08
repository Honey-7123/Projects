from prophet import Prophet
from prepare_data import load_and_merge_data
def train_prophet(store_id=1,dept_id=1,periods=90):
    df=load_and_merge_data()
    df=df[(df["Store"]==store_id)&(df["Dept"]==dept_id)]
    prophet_df=df[["Date","Weekly_Sales"]].rename(columns={"Date":"ds","Weekly_Sales":"y"})
    model=Prophet()
    model.fit(prophet_df)
    future=model.make_future_dataframe(periods=periods)
    forecast=model.predict(future)
    return forecast[["ds","yhat","yhat_lower","yhat_upper"]].tail(periods)
