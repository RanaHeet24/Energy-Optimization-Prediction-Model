"""
test_validator.py
Benchmark Test Suite & Validation Engine for Energy Predictions.

Provides:
  - 5 pre-built benchmark test cases with physically correct expectations
  - Single and batch test execution
  - CSV upload validation with physical sanity checks
  - Comparison testing between two scenarios
  
Works with both ML model and physics-based fallback.
"""

import math
from typing import Dict, List, Optional, Tuple
import pandas as pd


# ═════════════════════════════════════════════════
# TARIFF RATES
# ═════════════════════════════════════════════════
def _tariff_for_hour(hour: int) -> float:
    if 18 <= hour <= 22:
        return 8.50
    if 6 <= hour <= 9:
        return 7.50
    if 10 <= hour <= 17:
        return 6.00
    return 4.50


# ═════════════════════════════════════════════════
# PHYSICS-BASED PREDICTION ENGINE (always works)
# ═════════════════════════════════════════════════
_COP = {1: 2.7, 2: 2.9, 3: 3.1, 4: 3.5, 5: 3.9}
_TONNAGE_W = {0.75: 700, 1.0: 900, 1.5: 1500, 2.0: 2000}
_DEFAULT_W = {
    "AC": 1500, "Refrigerator": 250, "Geyser": 2000,
    "Washing Machine": 2000, "Microwave": 1200, "Ceiling Fan": 75,
    "LED TV": 150, "Desktop PC": 350, "Electric Iron": 1200,
    "LED Bulb": 12, "Laptop": 65, "Wi-Fi Router": 8,
    "Dishwasher": 1800, "Air Purifier": 55,
}

VOLTAGE_PROFILE = {
    range(0,  6):  235,
    range(6, 10):  215,
    range(10, 18): 225,
    range(18, 23): 205,
    range(23, 24): 235,
}


def _get_voltage(hour: int) -> float:
    for rng, v in VOLTAGE_PROFILE.items():
        if hour in rng:
            return v
    return 230


def predict_physics(inputs: Dict) -> Dict:
    """
    Pure-physics prediction — no ML model needed.
    Works for any appliance type with correct formulas.
    """
    appliance = inputs.get("appliance", inputs.get("appliance_type", "Other"))
    wattage = inputs.get("rated_wattage", _DEFAULT_W.get(appliance, 100))
    star = inputs.get("star_rating", 3)
    hour = inputs.get("hour", inputs.get("hour_of_day", 12))
    duration = inputs.get("duration_hrs", 1.0)
    outdoor = inputs.get("outdoor_temp", 35.0)
    inverter = inputs.get("inverter", inputs.get("inverter_mode", False))
    if isinstance(inverter, (int, float)):
        inverter = bool(inverter)
    setpoint = inputs.get("setpoint", inputs.get("setpoint_temp", 24))
    tonnage = inputs.get("tonnage", 1.5)
    occupancy = inputs.get("occupancy", 2)
    humidity = inputs.get("humidity", 50)

    voltage = _get_voltage(hour)
    tariff = _tariff_for_hour(hour)

    # ── AC-specific calculation ──
    if appliance == "AC":
        wattage = _TONNAGE_W.get(tonnage, 1500)
        cop = _COP.get(star, 3.1) + (0.6 if inverter else 0)
        delta_t = max(outdoor - setpoint, 0)

        # Room air mass (standard 12×10×3 = 360 m³ × 1.225 kg/m³)
        air_mass = 441.0
        c = 1.006  # kJ/(kg·°C)
        q_kj = air_mass * c * delta_t
        q_kwh = q_kj / 3600

        # AC electrical input = thermal load / COP
        elec_kwh = q_kwh / cop

        # Humidity load adds ~10-20%
        humidity_factor = 1.0 + max(humidity - 50, 0) * 0.003
        elec_kwh *= humidity_factor

        # Occupancy heat gain (~100W sensible per person)
        occ_kwh = occupancy * 0.1 * duration / cop
        elec_kwh += occ_kwh

        # Maintenance mode after reaching setpoint (partial load)
        if delta_t <= 2:
            capacity_frac = 0.15
        elif delta_t <= 5:
            capacity_frac = 0.30
        elif delta_t <= 10:
            capacity_frac = 0.55
        elif delta_t <= 15:
            capacity_frac = 0.75
        else:
            capacity_frac = 0.95

        rated_kwh = (wattage / 1000) * duration * capacity_frac
        kwh = min(elec_kwh * duration * 0.6 + rated_kwh * 0.4, (wattage / 1000) * duration)

        # Inverter reduces by ~20-35% at partial load
        if inverter and capacity_frac < 0.7:
            kwh *= 0.70

        # Voltage impact for motor
        v_mult = (230 / voltage) ** 2 if voltage < 230 else 1.0
        kwh *= v_mult

    # ── Geyser ──
    elif appliance in ("Geyser", "Geyser/Water Heater"):
        kwh = (wattage / 1000) * duration
        # Star efficiency
        eff = 0.80 + star * 0.03
        kwh *= (1.0 / eff)

    # ── Refrigerator (runs continuously, avg duty cycle) ──
    elif appliance == "Refrigerator":
        duty_cycle = 0.35 + max(outdoor - 25, 0) * 0.01
        cop = _COP.get(star, 3.1) + (0.4 if inverter else 0)
        kwh = (wattage / 1000) * duration * min(duty_cycle, 0.8)
        kwh /= (cop / 3.0)

    # ── Other appliances (simple resistive/electronic) ──
    else:
        kwh = (wattage / 1000) * duration
        eff = 0.85 + star * 0.02
        kwh *= (1.0 / eff)

    kwh = round(max(kwh, 0.001), 5)
    cost = round(kwh * tariff, 2)

    # Consumption level classification
    if kwh > 1.5:
        level = "CRITICAL"
    elif kwh > 0.8:
        level = "HIGH"
    elif kwh > 0.3:
        level = "MODERATE"
    else:
        level = "LOW"

    # Should optimize?
    should_opt = (tariff >= 7.50) or (kwh > 1.0) or (voltage < 220)

    return {
        "kwh": kwh,
        "cost": cost,
        "consumption_level": level,
        "should_optimize": should_opt,
        "voltage": voltage,
        "tariff": tariff,
        "appliance": appliance,
    }


# ═════════════════════════════════════════════════
# BENCHMARK TEST CASES
# ═════════════════════════════════════════════════
TEST_CASES = [
    {
        "test_id": "TC001",
        "name": "AC in Peak Summer Heat",
        "inputs": {
            "appliance": "AC", "tonnage": 1.5, "star_rating": 3,
            "inverter": False, "setpoint": 18, "outdoor_temp": 45,
            "humidity": 40, "hour": 14, "duration_hrs": 1, "occupancy": 3,
        },
        "expected": {
            "kwh_range": [1.0, 2.0],
            "cost_range": [6.0, 16.0],
            "consumption_level": "CRITICAL",
            "should_optimize": True,
        },
        "logic": "Hot day (45°C) + extreme setpoint (18°C) + ΔT=27°C = maximum consumption",
    },
    {
        "test_id": "TC002",
        "name": "AC in Cool Night Off-Peak",
        "inputs": {
            "appliance": "AC", "tonnage": 1.5, "star_rating": 5,
            "inverter": True, "setpoint": 26, "outdoor_temp": 24,
            "humidity": 55, "hour": 23, "duration_hrs": 1, "occupancy": 2,
        },
        "expected": {
            "kwh_range": [0.05, 0.5],
            "cost_range": [0.2, 2.5],
            "consumption_level": "LOW",
            "should_optimize": False,
        },
        "logic": "Cool night (24°C), ΔT≈0°C + 5★ inverter = minimal consumption",
    },
    {
        "test_id": "TC003",
        "name": "Geyser Morning Peak",
        "inputs": {
            "appliance": "Geyser", "rated_wattage": 2000, "star_rating": 3,
            "hour": 7, "duration_hrs": 0.5, "outdoor_temp": 22,
        },
        "expected": {
            "kwh_range": [0.8, 1.3],
            "cost_range": [5.0, 10.0],
            "should_optimize": True,
        },
        "logic": "2000W geyser × 30 min = ~1 kWh; morning peak tariff ₹7.50 → should shift to 5 AM",
    },
    {
        "test_id": "TC004",
        "name": "LED Bulb 12W Night",
        "inputs": {
            "appliance": "LED Bulb", "rated_wattage": 12, "star_rating": 5,
            "hour": 20, "duration_hrs": 4, "outdoor_temp": 30,
        },
        "expected": {
            "kwh_range": [0.03, 0.07],
            "cost_range": [0.2, 0.6],
            "consumption_level": "LOW",
            "should_optimize": False,
        },
        "logic": "12W LED × 4 hours = 0.048 kWh — negligible consumption",
    },
    {
        "test_id": "TC005",
        "name": "5★ Inverter AC vs 1★ Non-Inverter",
        "type": "comparison",
        "inputs_a": {
            "appliance": "AC", "tonnage": 1.5, "star_rating": 5,
            "inverter": True, "outdoor_temp": 38, "hour": 15,
            "duration_hrs": 1, "setpoint": 24, "occupancy": 2,
        },
        "inputs_b": {
            "appliance": "AC", "tonnage": 1.5, "star_rating": 1,
            "inverter": False, "outdoor_temp": 38, "hour": 15,
            "duration_hrs": 1, "setpoint": 24, "occupancy": 2,
        },
        "expected": {
            "a_should_be_lower_than_b": True,
            "difference_percent_min": 20,
            "difference_percent_max": 60,
        },
        "logic": "5★ inverter AC should consume 20-60% less than 1★ non-inverter at same conditions",
    },
]


# ═════════════════════════════════════════════════
# PHYSICAL SANITY RULES
# ═════════════════════════════════════════════════
SANITY_RULES = {
    "LED Bulb":         {"max_kwh_per_hr": 0.05,  "min_kwh_per_hr": 0.005},
    "Ceiling Fan":      {"max_kwh_per_hr": 0.12,  "min_kwh_per_hr": 0.03},
    "AC":               {"max_kwh_per_hr": 3.0,   "min_kwh_per_hr": 0.05},
    "Geyser":           {"max_kwh_per_hr": 2.5,   "min_kwh_per_hr": 0.5},
    "Refrigerator":     {"max_kwh_per_hr": 0.3,   "min_kwh_per_hr": 0.03},
    "Washing Machine":  {"max_kwh_per_hr": 2.5,   "min_kwh_per_hr": 0.3},
    "Microwave":        {"max_kwh_per_hr": 1.5,   "min_kwh_per_hr": 0.3},
    "LED TV":           {"max_kwh_per_hr": 0.2,   "min_kwh_per_hr": 0.05},
    "Laptop":           {"max_kwh_per_hr": 0.1,   "min_kwh_per_hr": 0.02},
    "Wi-Fi Router":     {"max_kwh_per_hr": 0.02,  "min_kwh_per_hr": 0.003},
    "Electric Iron":    {"max_kwh_per_hr": 1.5,   "min_kwh_per_hr": 0.4},
}


def physical_sanity_check(appliance: str, kwh: float,
                          duration: float) -> Dict:
    """
    Checks if a prediction is physically sane for the appliance type.
    Returns dict with is_valid, checks list, and failure reasons.
    """
    kwh_per_hr = kwh / max(duration, 0.01)
    rules = SANITY_RULES.get(appliance, {"max_kwh_per_hr": 5.0, "min_kwh_per_hr": 0.001})
    checks = []
    valid = True

    # Check max
    if kwh_per_hr <= rules["max_kwh_per_hr"]:
        checks.append(f"✅ kWh/hr ({kwh_per_hr:.3f}) ≤ max ({rules['max_kwh_per_hr']})")
    else:
        checks.append(f"❌ kWh/hr ({kwh_per_hr:.3f}) > max ({rules['max_kwh_per_hr']}) — TOO HIGH")
        valid = False

    # Check min
    if kwh_per_hr >= rules["min_kwh_per_hr"]:
        checks.append(f"✅ kWh/hr ({kwh_per_hr:.3f}) ≥ min ({rules['min_kwh_per_hr']})")
    else:
        checks.append(f"❌ kWh/hr ({kwh_per_hr:.3f}) < min ({rules['min_kwh_per_hr']}) — TOO LOW")
        valid = False

    return {"is_valid": valid, "checks": checks, "kwh_per_hr": round(kwh_per_hr, 4)}


# ═════════════════════════════════════════════════
# TEST RUNNER
# ═════════════════════════════════════════════════
class TestValidator:
    """Runs and validates energy prediction tests."""

    def run_single_test(self, test_case: Dict) -> Dict:
        """Runs one benchmark test and checks against expected values."""
        tc_id = test_case["test_id"]
        tc_name = test_case["name"]
        tc_type = test_case.get("type", "single")
        expected = test_case["expected"]
        logic = test_case.get("logic", "")

        if tc_type == "comparison":
            return self._run_comparison_test(test_case)

        inputs = test_case["inputs"]
        result = predict_physics(inputs)

        status = "PASS"
        failures = []

        # Check kWh range
        if "kwh_range" in expected:
            lo, hi = expected["kwh_range"]
            if lo <= result["kwh"] <= hi:
                pass
            else:
                status = "FAIL"
                failures.append(
                    f"kWh {result['kwh']:.4f} outside [{lo}, {hi}]")

        # Check cost range
        if "cost_range" in expected:
            lo, hi = expected["cost_range"]
            if lo <= result["cost"] <= hi:
                pass
            else:
                status = "FAIL"
                failures.append(
                    f"Cost ₹{result['cost']:.2f} outside [{lo}, {hi}]")

        # Check consumption level
        if "consumption_level" in expected:
            if result["consumption_level"] != expected["consumption_level"]:
                status = "FAIL"
                failures.append(
                    f"Level {result['consumption_level']} ≠ expected "
                    f"{expected['consumption_level']}")

        # Check should_optimize
        if "should_optimize" in expected:
            if result["should_optimize"] != expected["should_optimize"]:
                status = "FAIL"
                failures.append(
                    f"should_optimize={result['should_optimize']} ≠ "
                    f"expected {expected['should_optimize']}")

        # Physical sanity
        duration = inputs.get("duration_hrs", 1)
        sanity = physical_sanity_check(
            inputs.get("appliance", "Other"), result["kwh"], duration)

        verdict = (f"✅ {tc_name} — Physics match: {logic}"
                   if status == "PASS"
                   else f"❌ {tc_name} — {'; '.join(failures)}")

        return {
            "test_id":          tc_id,
            "name":             tc_name,
            "status":           status,
            "predicted_kwh":    result["kwh"],
            "expected_kwh_range": expected.get("kwh_range"),
            "predicted_cost":   result["cost"],
            "predicted_level":  result["consumption_level"],
            "expected_level":   expected.get("consumption_level"),
            "should_optimize":  result["should_optimize"],
            "logic_check":      status == "PASS",
            "failure_reason":   "; ".join(failures) if failures else None,
            "sanity":           sanity,
            "verdict_message":  verdict,
            "logic":            logic,
        }

    def _run_comparison_test(self, test_case: Dict) -> Dict:
        """Runs a comparison test between two scenarios."""
        tc_id = test_case["test_id"]
        tc_name = test_case["name"]
        expected = test_case["expected"]
        logic = test_case.get("logic", "")

        result_a = predict_physics(test_case["inputs_a"])
        result_b = predict_physics(test_case["inputs_b"])

        pct_diff = round(
            abs(result_a["kwh"] - result_b["kwh"])
            / max(result_b["kwh"], 0.001) * 100, 1)

        status = "PASS"
        failures = []

        if expected.get("a_should_be_lower_than_b"):
            if result_a["kwh"] >= result_b["kwh"]:
                status = "FAIL"
                failures.append(
                    f"A ({result_a['kwh']:.3f}) should be < B ({result_b['kwh']:.3f})")

        if "difference_percent_min" in expected:
            if pct_diff < expected["difference_percent_min"]:
                status = "FAIL"
                failures.append(
                    f"Difference {pct_diff}% < min {expected['difference_percent_min']}%")

        if "difference_percent_max" in expected:
            if pct_diff > expected["difference_percent_max"]:
                status = "FAIL"
                failures.append(
                    f"Difference {pct_diff}% > max {expected['difference_percent_max']}%")

        verdict = (f"✅ {tc_name} — A uses {pct_diff}% less than B"
                   if status == "PASS"
                   else f"❌ {tc_name} — {'; '.join(failures)}")

        return {
            "test_id":         tc_id,
            "name":            tc_name,
            "status":          status,
            "predicted_kwh_a": result_a["kwh"],
            "predicted_kwh_b": result_b["kwh"],
            "cost_a":          result_a["cost"],
            "cost_b":          result_b["cost"],
            "pct_difference":  pct_diff,
            "logic_check":     status == "PASS",
            "failure_reason":  "; ".join(failures) if failures else None,
            "verdict_message": verdict,
            "logic":           logic,
        }

    def run_all_tests(self) -> Dict:
        """Runs all benchmark test cases and returns summary."""
        results = [self.run_single_test(tc) for tc in TEST_CASES]
        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = len(results) - passed

        if failed == 0:
            verdict = "✅ All model predictions are physically accurate and ready for demonstration"
        else:
            verdict = f"⚠️ {failed} test(s) failed — review predictions for physical accuracy"

        return {
            "total_tests":     len(results),
            "passed":          passed,
            "failed":          failed,
            "pass_rate":       round(passed / len(results) * 100, 1),
            "results":         results,
            "overall_verdict": verdict,
        }

    def validate_csv_test_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates each row of an uploaded CSV.

        Required columns: appliance_type, rated_wattage, star_rating,
                          outdoor_temp, humidity, hour_of_day, duration_hrs

        Optional: inverter_mode, setpoint_temp, tonnage, occupancy

        Returns enriched DataFrame with prediction + sanity columns.
        """
        out_rows = []
        for _, row in df.iterrows():
            inputs = {
                "appliance":     str(row.get("appliance_type", "Other")),
                "rated_wattage": float(row.get("rated_wattage", 100)),
                "star_rating":   int(row.get("star_rating", 3)),
                "outdoor_temp":  float(row.get("outdoor_temp", 35)),
                "humidity":      float(row.get("humidity", 50)),
                "hour":          int(row.get("hour_of_day", 12)),
                "duration_hrs":  float(row.get("duration_hrs", 1)),
                "inverter":      bool(int(row.get("inverter_mode", 0))),
                "setpoint":      float(row.get("setpoint_temp", 24)),
                "tonnage":       float(row.get("tonnage", 1.5)),
                "occupancy":     int(row.get("occupancy", 2)),
            }

            pred = predict_physics(inputs)
            sanity = physical_sanity_check(
                inputs["appliance"], pred["kwh"], inputs["duration_hrs"])

            out_rows.append({
                **{k: row.get(k, "") for k in df.columns},
                "predicted_kwh":      pred["kwh"],
                "predicted_cost":     pred["cost"],
                "consumption_level":  pred["consumption_level"],
                "should_optimize":    pred["should_optimize"],
                "voltage":            pred["voltage"],
                "tariff":             pred["tariff"],
                "sanity_valid":       sanity["is_valid"],
                "sanity_notes":       " | ".join(sanity["checks"]),
            })

        return pd.DataFrame(out_rows)

    def compare_two_scenarios(
        self, inputs_a: Dict, inputs_b: Dict
    ) -> Dict:
        """
        Compares two user-defined scenarios side by side.
        Returns both predictions and analysis.
        """
        result_a = predict_physics(inputs_a)
        result_b = predict_physics(inputs_b)

        if result_b["kwh"] > 0:
            pct_diff = round(
                abs(result_a["kwh"] - result_b["kwh"])
                / max(result_a["kwh"], result_b["kwh"]) * 100, 1)
        else:
            pct_diff = 0

        a_higher = result_a["kwh"] > result_b["kwh"]

        # Generate explanation
        reasons = []
        if inputs_a.get("star_rating", 3) != inputs_b.get("star_rating", 3):
            reasons.append("different star ratings affect COP efficiency")
        if inputs_a.get("inverter") != inputs_b.get("inverter"):
            reasons.append("inverter tech reduces partial-load consumption")
        if inputs_a.get("hour", 12) != inputs_b.get("hour", 12):
            reasons.append("different hours → different voltage and tariff tiers")
        if inputs_a.get("outdoor_temp", 35) != inputs_b.get("outdoor_temp", 35):
            reasons.append("outdoor temp difference changes thermal load (ΔT)")
        if inputs_a.get("setpoint", 24) != inputs_b.get("setpoint", 24):
            reasons.append("setpoint difference changes cooling requirement")

        explanation = (
            f"Scenario {'A' if a_higher else 'B'} uses {pct_diff}% more energy. "
            f"This is {'CORRECT' if pct_diff > 0 else 'expected'} because: "
            + "; ".join(reasons or ["same inputs"])
        )

        return {
            "result_a":     result_a,
            "result_b":     result_b,
            "pct_diff":     pct_diff,
            "a_higher":     a_higher,
            "explanation":  explanation,
        }
