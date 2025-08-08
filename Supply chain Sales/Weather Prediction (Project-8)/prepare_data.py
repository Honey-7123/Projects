import pandas as pd

def load_and_preprocess_data():
    # Load the dataset
    df = pd.read_csv("AirQualityUCI.csv", sep=";", decimal=",", encoding="latin1")

    # Drop last two unnamed empty columns (common in this dataset)
    df = df.iloc[:, :-2]

    # Replace '.' with ':' in time so pandas can parse it
    df["Time"] = df["Time"].str.replace(".", ":", regex=False)

    # Combine Date and Time into a single datetime column
    df["DateTime"] = pd.to_datetime(df["Date"] + " " + df["Time"], dayfirst=True, errors='coerce')

    # Rename relevant column for modeling (we'll use CO concentration as target)
    df = df.rename(columns={"CO(GT)": "CO_concentration"})

    # Clean invalid values (-200 means missing in this dataset)
    df.replace(-200, pd.NA, inplace=True)
    df = df[["DateTime", "CO_concentration"]].dropna()
    df = df.ffill().bfill()

    return df
