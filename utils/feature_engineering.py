import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df_input: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Apply feature engineering to the raw dataset or inference data.
    Adds cyclical time encoding, weather interactions, appliance metrics,
    and lag/rolling features (if training).
    """
    df = df_input.copy()

    # Ensure timestamp is datetime
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # ─────────────────────────────────────────
    # TIME FEATURES (If 'timestamp' is present)
    # ─────────────────────────────────────────
    if 'timestamp' in df.columns:
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month

    # Peak hour flags
    df['is_peak_hour'] = df['hour_of_day'].apply(lambda h: 1 if 18 <= h <= 22 else 0)
    df['is_morning_peak'] = df['hour_of_day'].apply(lambda h: 1 if 6 <= h <= 9 else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)

    # Seasons: 0=Winter 1=Spring 2=Summer 3=Monsoon
    def get_season(m):
        if m in [1, 11, 12]: return 0
        elif m in [2, 3, 10]: return 1
        elif m in [4, 5, 6]: return 2
        else: return 3
    if 'month' in df.columns and 'season' not in df.columns:
        df['season'] = df['month'].apply(get_season)

    # ─────────────────────────────────────────
    # CYCLICAL TIME ENCODING
    # ─────────────────────────────────────────
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24.0)
    
    if 'month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    else:
        df['month_sin'] = 0.0
        df['month_cos'] = 1.0
        
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

    # ─────────────────────────────────────────
    # WEATHER INTERACTION FEATURES
    # ─────────────────────────────────────────
    df['thermal_delta'] = df['outdoor_temp'] - df['setpoint_temp']
    
    # Heat Index Formula (simplified version)
    # heat_index = temp + 0.33 * (humidity * 6.105 * exp(17.27 * temp / (237.7 + temp)) / 100) - 4
    e_term = np.exp(17.27 * df['outdoor_temp'] / (237.7 + df['outdoor_temp']))
    df['heat_index'] = df['outdoor_temp'] + 0.33 * (df['humidity'] * 6.105 * e_term / 100.0) - 4

    df['cooling_degree'] = df['outdoor_temp'].apply(lambda x: max(0, x - 18))
    df['heating_degree'] = df['outdoor_temp'].apply(lambda x: max(0, 18 - x))
    df['temp_humidity_index'] = df['outdoor_temp'] * (df['humidity'] / 100.0)

    # ─────────────────────────────────────────
    # APPLIANCE FEATURES
    # ─────────────────────────────────────────
    df['efficiency_score'] = df['star_rating'] * df['inverter_mode'].apply(lambda x: 1.1 if x == 1 else 1.0)
    
    # Age efficiency (assume 1 year if not present)
    df['age_years'] = df.get('age_years', 1.0)
    df['age_efficiency'] = df['age_years'].apply(lambda a: max(0.7, 1.0 - (a * 0.015)))

    # ─────────────────────────────────────────
    # VOLTAGE FEATURE
    # ─────────────────────────────────────────
    def get_voltage(hour):
        if 18 <= hour <= 22: return 205
        elif 6 <= hour <= 9: return 215
        elif 10 <= hour <= 17: return 225
        else: return 235

    if 'grid_voltage' not in df.columns:
        df['grid_voltage'] = df['hour_of_day'].apply(get_voltage)
        
    df['voltage_drop_percent'] = ((230.0 - df['grid_voltage']) / 230.0) * 100.0

    # ─────────────────────────────────────────
    # LAG FEATURES (Only for training data)
    # ─────────────────────────────────────────
    if is_training and 'power_kwh' in df.columns and 'timestamp' in df.columns:
        # Sort by appliance type and timestamp to prevent leaking data across appliances
        df = df.sort_values(by=['appliance_type', 'timestamp'])
        
        # Shift creates lag features
        df['lag_1'] = df.groupby('appliance_type')['power_kwh'].shift(1)
        df['lag_4'] = df.groupby('appliance_type')['power_kwh'].shift(4)
        df['lag_96'] = df.groupby('appliance_type')['power_kwh'].shift(96)
        
        # Rolling features
        df['rolling_mean_4'] = df.groupby('appliance_type')['power_kwh'].transform(lambda x: x.rolling(4, min_periods=1).mean())
        df['rolling_mean_96'] = df.groupby('appliance_type')['power_kwh'].transform(lambda x: x.rolling(96, min_periods=1).mean())
        df['rolling_std_4'] = df.groupby('appliance_type')['power_kwh'].transform(lambda x: x.rolling(4, min_periods=1).std())
        
        # Fill NA for first few rows
        df['lag_1'] = df.groupby('appliance_type')['lag_1'].bfill().fillna(0)
        df['lag_4'] = df.groupby('appliance_type')['lag_4'].bfill().fillna(0)
        df['lag_96'] = df.groupby('appliance_type')['lag_96'].bfill().fillna(0)
        df['rolling_std_4'] = df['rolling_std_4'].fillna(0)
        
        # Ensure it returns back to original time sort overall
        df = df.sort_values('timestamp').reset_index(drop=True)

    return df

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import LabelEncoder
    
    csv_path = "data/raw/appliance_energy_dataset.csv"
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded raw dataset shape: {df.shape}")
        
        df_eng = engineer_features(df, is_training=True)
        
        # Ensure we have appliance_encoded for correlation
        le = LabelEncoder()
        df_eng['appliance_encoded'] = le.fit_transform(df_eng['appliance_type'])
        
        target = "power_kwh"
        final_features = [
            "rated_wattage", "star_rating", "inverter_mode",
            "setpoint_temp", "outdoor_temp", "humidity",
            "thermal_delta", "heat_index", "cooling_degree",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
            "is_peak_hour", "is_morning_peak", "is_weekend",
            "season", "occupancy", "efficiency_score",
            "voltage_drop_percent", "appliance_encoded",
            "lag_1", "lag_4", "rolling_mean_4"
        ]
        
        corr_matrix = df_eng[final_features + [target]].corr()
        
        print("\n── Top 10 Features Correlated with power_kwh ──")
        top_corr = corr_matrix[target].abs().sort_values(ascending=False).drop(target)
        for col, val in top_corr.head(10).items():
            sign = "+" if corr_matrix[target][col] > 0 else "-"
            print(f"  {col:<20} : {sign} {val:.4f}")
            
        print(f"\nEngineered dataset shape: {df_eng.shape}")
        
    except Exception as e:
        print(f"Error: {e}")
