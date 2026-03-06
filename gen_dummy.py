import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random

random.seed(42)
np.random.seed(42)

DEVICES = ["AC", "Geyser", "Refrigerator", "Washing Machine", "Microwave",
           "Ceiling Fan", "LED TV", "Desktop PC", "Electric Iron", "LED Bulb",
           "Laptop", "Wi-Fi Router"]

start_dt = datetime(2023, 1, 1, 0, 0)
rows = []

for _ in range(8500):
    device = random.choice(DEVICES)
    day_offset = random.randint(0, 364)
    hour = random.randint(0, 23)
    minute = random.choice([0, 15, 30, 45])
    ts = start_dt + timedelta(days=day_offset, hours=hour, minutes=minute)

    # Temperature (Gujarat climate)
    month = ts.month
    if month in [4, 5, 6]:
        base_temp = np.random.uniform(32, 43)
    elif month in [7, 8, 9]:
        base_temp = np.random.uniform(26, 35)
    elif month in [11, 12, 1]:
        base_temp = np.random.uniform(12, 22)
    else:
        base_temp = np.random.uniform(22, 32)

    temperature = round(float(base_temp + np.random.normal(0, 1.5)), 1)

    # Occupancy
    is_weekend = 1 if ts.weekday() >= 5 else 0
    if is_weekend:
        occupancy = random.randint(2, 5)
    elif 9 <= hour <= 17:
        occupancy = random.randint(0, 2)
    elif 18 <= hour <= 22:
        occupancy = random.randint(2, 5)
    else:
        occupancy = random.randint(1, 4)

    # Energy (kWh for 15-min slot)
    watt_map = {
        "AC": 1500, "Geyser": 2000, "Refrigerator": 250,
        "Washing Machine": 1000, "Microwave": 1200, "Ceiling Fan": 60,
        "LED TV": 100, "Desktop PC": 350, "Electric Iron": 1200,
        "LED Bulb": 10, "Laptop": 55, "Wi-Fi Router": 8,
    }
    rated_w = watt_map[device]
    usage_factor = round(float(np.random.uniform(0.5, 0.95)), 3)
    energy_kwh = round((rated_w * usage_factor * 0.25) / 1000, 5)

    rows.append({
        "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "energy_kwh": energy_kwh,
        "temperature": temperature,
        "occupancy": occupancy,
    })

df = pd.DataFrame(rows)
df = df.sort_values("timestamp").reset_index(drop=True)

os.makedirs(r"d:\CLUADE VALA PROJECT\data\raw", exist_ok=True)
out_path = r"d:\CLUADE VALA PROJECT\data\raw\appliance_energy_dataset.csv"
df.to_csv(out_path, index=False)
print(f"[SAVED] {out_path}  ({len(df)} rows)")
print(df.head(5).to_string())
