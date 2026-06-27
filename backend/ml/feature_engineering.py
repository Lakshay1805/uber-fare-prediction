import pandas as pd
import numpy as np



def invalid_coordinates_cleaning(df):
    df = df[
        (df["pickup_latitude"].between(-90, 90))
        & (df["dropoff_latitude"].between(-90, 90))
        & (df["pickup_longitude"].between(-180, 180))
        & (df["dropoff_longitude"].between(-180, 180))
    ]
    df = df[
        (df["pickup_latitude"] != 0)
        & (df["pickup_longitude"] != 0)
        & (df["dropoff_latitude"] != 0)
        & (df["dropoff_longitude"] != 0)
    ]
    return df


def invalid_passenger_cleaning(df):
    df["passenger_count"] = df["passenger_count"].clip(1, 6)
    return df


def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def distance_calculation(df):
    df["distance_km"] = df.apply(
        lambda row: haversine(
            row["pickup_latitude"],
            row["pickup_longitude"],
            row["dropoff_latitude"],
            row["dropoff_longitude"],
        ),
        axis=1,
    )

    return df


def datetime_extraction(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["hour"] = df["pickup_datetime"].dt.hour
    df["year"] = df["pickup_datetime"].dt.year
    df["month"] = df["pickup_datetime"].dt.month
    df["weekday"] = df["pickup_datetime"].dt.weekday
    df["monthly_quarters"] = df["month"].map(
        {
            1: "Q1",
            2: "Q1",
            3: "Q1",
            4: "Q2",
            5: "Q2",
            6: "Q2",
            7: "Q3",
            8: "Q3",
            9: "Q3",
            10: "Q4",
            11: "Q4",
            12: "Q4",
        }
    )
    df["Hourly_Segments"] = df.hour.map(
        {
            0: "H1",
            1: "H1",
            2: "H1",
            3: "H1",
            4: "H2",
            5: "H2",
            6: "H2",
            7: "H2",
            8: "H3",
            9: "H3",
            10: "H3",
            11: "H3",
            12: "H4",
            13: "H4",
            14: "H4",
            15: "H4",
            16: "H5",
            17: "H5",
            18: "H5",
            19: "H5",
            20: "H6",
            21: "H6",
            22: "H6",
            23: "H6",
        }
    )
    df["is_peakhour"] = df["hour"].apply(peak_hour)

    return df


def peak_hour(hour):
    if 7 <= hour <= 10 or 17 <= hour <= 20:
        return 1
    return 0


def fare_outliers_removal(df):
    lower = df["fare_amount"].quantile(0.01)
    upper = df["fare_amount"].quantile(0.99)

    df = df[(df["fare_amount"] >= lower) & (df["fare_amount"] <= upper)]

    return df


def distance_outliers_removal(df):
    lower = df["distance_km"].quantile(0.01)
    upper = df["distance_km"].quantile(0.99)

    df = df[(df["distance_km"] >= lower) & (df["distance_km"] <= upper)]

    return df


def features_selection(df):
    df.drop(columns=["Unnamed: 0", "key", "hour", "month"],errors="ignore", inplace=True)

    return df


def raw_data_cleaning(df):
    df = df.dropna()    # only 2 rows contains null values so we are dropping it.
    df = invalid_coordinates_cleaning(df)
    df = invalid_passenger_cleaning(df)

    return df


def feature_creation(df):
    df = distance_calculation(df)
    df = datetime_extraction(df)

    return df


def featured_data_cleaning(df):
    df = fare_outliers_removal(df)
    df = distance_outliers_removal(df)
    df = features_selection(df)

    return df


def full_feature_pipeline(df):
    df = raw_data_cleaning(df)
    df = feature_creation(df)
    df = featured_data_cleaning(df)
    return df


def predict_feature_pipeline(df):
    df = raw_data_cleaning(df)
    df = feature_creation(df)
    df = features_selection(df)
    return df