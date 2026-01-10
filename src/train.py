import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import preprocess_data

def train_model():
    # Preprocess data
    df, feature_cols = preprocess_data()
    # Features and target
    X = df[feature_cols]
    y = df['no_show']
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print(f"Training set size: {X_train_scaled.shape[0]} samples")
    print(f"Validation set size: {X_val_scaled.shape[0]} samples")
    print(f"Number of features: {X_train_scaled.shape[1]}")
    # Train Random Forest
    rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced')
    rf.fit(X_train_scaled, y_train)
    # Return model and validation data
    return rf, X_val_scaled, y_val