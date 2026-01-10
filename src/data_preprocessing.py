import pandas as pd
import numpy as np

def preprocess_data():
    # Load raw data
    df = pd.read_csv("data/raw/KaggleV2-May-2016.csv")

    # Target
    df['no_show'] = df['No-show'].map({'No': 0, 'Yes': 1})

    # Dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # Days between scheduling and appointment
    df['days_between'] = (
        df['AppointmentDay'].dt.normalize()
        - df['ScheduledDay'].dt.normalize()
    ).dt.days

    # Remove rows with negative days_between
    df = df[df['days_between'] >= 0]

    df['appointment_weekday'] = df['AppointmentDay'].dt.weekday
    df['is_weekend'] = df['appointment_weekday'].isin([5, 6]).astype(int)

    # Demographics
    df['gender'] = df['Gender'].map({'F': 0, 'M': 1})
    df.loc[df['Age'] < 0, 'Age'] = np.nan

    # Neighborhood frequency encoding
    neighborhood_freq = df['Neighbourhood'].value_counts(normalize=True)
    df['neighborhood_freq'] = df['Neighbourhood'].map(neighborhood_freq)

    feature_cols = [
        'Age', 'gender', 'days_between', 'appointment_weekday', 'is_weekend',
        'SMS_received', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap',
        'neighborhood_freq'
    ]

    df = df.dropna(subset=feature_cols)
    return df, feature_cols