"""
utils/predictor.py
15-minute slot energy predictor using trained XGBoost model.
Falls back to physics-based demo if model not found.
"""

import os, json
import numpy as np
from typing import Optional
from datetime import datetime, timedelta
from voltage_calculator import VoltageDropCalculator

try:
    import joblib
    JOBLIB_OK = True
except ImportError:
    JOBLIB_OK = False

MODEL_DIR  = "models/saved"
FEATURES   = [
    "rated_wattage", "star_rating", "inverter_mode",
    "setpoint_temp", "outdoor_temp", "humidity",
    "thermal_delta", "hour_of_day", "is_peak_hour",
    "day_of_week", "is_weekend", "occupancy", "usage_factor",
]


# ──────────────────────────────────────────────
# LOAD SAVED MODEL (returns None if not found)
# ──────────────────────────────────────────────
def _load_model():
    try:
        model  = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
        scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
        return model, scaler
    except Exception:
        return None, None


# ──────────────────────────────────────────────
# PHYSICS FALLBACK (when model not available)
# ──────────────────────────────────────────────
def _physics_predict(rated_w: int, star: int, inverter: int,
                     setpoint: float, outdoor_temp: float,
                     humidity: float) -> float:
    if setpoint and outdoor_temp:
        td = outdoor_temp - setpoint
        uf = (0.60 + td * 0.015 + humidity * 0.001
              - (star - 3) * 0.05 - inverter * 0.10)
        uf = float(np.clip(uf, 0.30, 1.00))
    else:
        uf = np.random.uniform(0.70, 0.90)
    kwh = (rated_w * uf * 0.25) / 1000
    return round(float(np.clip(kwh + np.random.normal(0, kwh * 0.03), 0.0001, 5.0)), 5)


# ──────────────────────────────────────────────
# INTERPOLATE WEATHER ACROSS SLOTS
# ──────────────────────────────────────────────
def _interp_weather(slot_i: int, total_slots: int,
                    weather: dict, forecast: Optional[list] = None) -> tuple:
    """Returns (temp, humidity) for a given slot index."""
    if forecast and len(forecast) > slot_i:
        return forecast[slot_i]["temp"], forecast[slot_i]["humidity"]

    # Slight drift: temp dips slightly later in evening
    base_t = weather.get("outdoor_temp", 32)
    base_h = weather.get("humidity", 55)
    hour_now = datetime.now().hour

    drift_t = -0.15 * slot_i if hour_now >= 14 else 0.10 * slot_i
    drift_h =  0.20 * slot_i

    t = round(float(np.clip(base_t + drift_t + np.random.normal(0, 0.2), -5, 50)), 1)
    h = round(float(np.clip(base_h + drift_h + np.random.normal(0, 0.5), 5, 99)), 1)
    return t, h


# ──────────────────────────────────────────────
# USAGE FACTOR (same physics as dataset gen)
# ──────────────────────────────────────────────
def _calc_uf(app: str, rated_w: int, star: int, inverter: int,
             setpoint: float, outdoor_temp: float, humidity: float) -> float:
    if app == "AC":
        td = outdoor_temp - setpoint
        uf = 0.60 + td * 0.015 + humidity * 0.001 - (star - 3) * 0.05 - inverter * 0.10
        return float(np.clip(uf, 0.30, 1.00))
    elif app == "Refrigerator":
        return float(np.clip(0.65 + (outdoor_temp - 25) * 0.005 - (star - 3) * 0.03, 0.4, 0.95))
    elif app == "Geyser":
        return float(np.clip(0.85 - (outdoor_temp - 15) * 0.008, 0.5, 1.0))
    else:
        return float(np.clip(np.random.uniform(0.72, 0.92), 0.5, 1.0))


# ──────────────────────────────────────────────
# MAIN PREDICTION FUNCTION
# ──────────────────────────────────────────────
def predict_session(
    appliance_type:  str,
    rated_wattage:   int,
    star_rating:     int,
    inverter_mode:   int,
    setpoint_temp:   float,
    start_hour:      int,
    start_minute:    int,
    duration_hours:  float,
    weather_data:    dict,
    occupancy:       int,
    forecast:        Optional[list] = None,
) -> dict:
    """
    Predicts energy usage for every 15-minute slot of an appliance session.
    Uses XGBoost model if available, else physics fallback.
    """
    model, scaler = _load_model() if JOBLIB_OK else (None, None)
    demo_mode = (model is None or scaler is None)
    voltage_calc = VoltageDropCalculator()

    total_slots = int(duration_hours * 4)   # 1 hr = 4 slots
    slots       = []
    cum_kwh     = 0.0
    cum_cost    = 0.0

    # Start datetime
    now = datetime.now()
    slot_dt = now.replace(hour=start_hour, minute=start_minute, second=0, microsecond=0)
    dow = slot_dt.weekday()
    is_weekend = 1 if dow >= 5 else 0

    for i in range(total_slots):
        current_hour   = slot_dt.hour
        current_minute = slot_dt.minute
        is_peak        = 1 if 9 <= current_hour <= 22 else 0
        tariff         = 8.50 if is_peak else 4.50

        temp, hum = _interp_weather(i, total_slots, weather_data, forecast)
        sp = setpoint_temp if setpoint_temp else temp
        td = round(temp - sp, 2)
        uf = _calc_uf(appliance_type, rated_wattage, star_rating,
                      inverter_mode, sp, temp, hum)

        if not demo_mode:
            try:
                feat_vec = [[
                    rated_wattage, star_rating, inverter_mode,
                    sp, temp, hum, td,
                    current_hour, is_peak, dow, is_weekend,
                    occupancy, uf,
                ]]
                kwh = float(scaler and model and
                            model.predict(scaler.transform(feat_vec))[0])
                kwh = round(float(np.clip(kwh, 0.0001, 5.0)), 5)
            except Exception:
                kwh = _physics_predict(rated_wattage, star_rating, inverter_mode,
                                       sp, temp, hum)
                demo_mode = True
        else:
            kwh = _physics_predict(rated_wattage, star_rating, inverter_mode,
                                   sp, temp, hum)

        # ── Voltage Drop Impact (BEE India / IEEE research) ──
        v_impact = voltage_calc.calculate_voltage_impact(
            appliance_type, rated_wattage, current_hour
        )
        kwh_base = kwh
        kwh = round(kwh * v_impact["multiplier"], 5)
        extra_kwh_voltage = round(kwh - kwh_base, 5)

        cost     = round(kwh * tariff, 4)
        cum_kwh  = round(cum_kwh + kwh, 5)
        cum_cost = round(cum_cost + cost, 4)

        slots.append({
            "slot":            i + 1,
            "time":            slot_dt.strftime("%H:%M"),
            "hour":            current_hour,
            "outdoor_temp":    temp,
            "humidity":        hum,
            "thermal_delta":   td,
            "usage_factor":    round(uf, 4),
            "kwh_base":        kwh_base,
            "kwh":             kwh,
            "extra_kwh_voltage": extra_kwh_voltage,
            "voltage":         v_impact["grid_voltage"],
            "voltage_multiplier": v_impact["multiplier"],
            "cost_inr":        cost,
            "tariff":          tariff,
            "is_peak":         bool(is_peak),
            "cumulative_kwh":  cum_kwh,
            "cumulative_cost": cum_cost,
        })
        slot_dt += timedelta(minutes=15)

    total_kwh  = round(sum(s["kwh"] for s in slots), 4)
    total_cost = round(sum(s["cost_inr"] for s in slots), 4)
    avg_watts  = round((total_kwh / duration_hours) * 1000, 1) if duration_hours else 0
    peak_slot  = max(slots, key=lambda s: s["kwh"])
    co2_kg     = round(total_kwh * 0.82, 4)  # India emission factor

    # ── Voltage impact totals ──
    voltage_extra_kwh  = round(sum(s["extra_kwh_voltage"] for s in slots), 4)
    voltage_extra_cost = round(sum(
        s["extra_kwh_voltage"] * s["tariff"] for s in slots
    ), 4)
    # Representative voltage impact for the session (use first slot's hour)
    session_voltage_impact = voltage_calc.calculate_voltage_impact(
        appliance_type, rated_wattage, start_hour
    )

    return {
        "slots":               slots,
        "total_kwh":           total_kwh,
        "total_cost_inr":      total_cost,
        "avg_watts":           avg_watts,
        "peak_slot":           {"time": peak_slot["time"], "kwh": peak_slot["kwh"]},
        "session_duration_hrs": duration_hours,
        "co2_kg":              co2_kg,
        "demo_mode":           demo_mode,
        "appliance":           appliance_type,
        "start_time":          f"{start_hour:02d}:{start_minute:02d}",
        "voltage_extra_kwh":   voltage_extra_kwh,
        "voltage_extra_cost":  voltage_extra_cost,
        "voltage_impact":      session_voltage_impact,
    }
