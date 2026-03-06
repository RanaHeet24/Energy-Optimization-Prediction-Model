"""
home_profile.py
Home Appliance Profile Manager.

Manages a user's complete home appliance inventory with:
- Per-appliance specs (wattage, star rating, inverter, age)
- Weekly usage patterns (hour-by-hour per day)
- Monthly kWh and cost estimation
- Peak hour analysis for optimization recommendations
- Save/load JSON and CSV export
"""

import json
import os
import uuid
from typing import Dict, List, Optional
from datetime import datetime


# ── Default wattages by appliance type ────────────────
DEFAULT_WATTAGES = {
    "AC":               1500,
    "Refrigerator":     250,
    "Geyser":           2000,
    "Washing Machine":  2000,
    "Microwave":        1200,
    "Ceiling Fan":      75,
    "LED TV":           150,
    "Desktop PC":       350,
    "Electric Iron":    1200,
    "LED Bulb":         12,
    "Laptop":           65,
    "Wi-Fi Router":     8,
    "Dishwasher":       1800,
    "Air Purifier":     55,
    "Other":            100,
}

PEAK_HOURS = list(range(9, 23))  # 9 AM – 10 PM
PEAK_TARIFF    = 8.50   # ₹/kWh
OFFPEAK_TARIFF = 4.50   # ₹/kWh
DAYS_OF_WEEK   = ["monday", "tuesday", "wednesday", "thursday",
                  "friday", "saturday", "sunday"]


def get_default_wattage(appliance_type: str) -> int:
    """Returns default wattage for a given appliance type."""
    return DEFAULT_WATTAGES.get(appliance_type, 100)


def is_peak_hour(hour: int) -> bool:
    """Returns True if hour is during peak tariff period (9 AM – 10 PM)."""
    return hour in PEAK_HOURS


class HomeProfile:
    """
    Manages a user's home appliance inventory.

    Usage:
        profile = HomeProfile()
        profile.add_appliance({...})
        total = profile.calculate_monthly_bill()
        profile.save_to_json("home_profile.json")
    """

    def __init__(self) -> None:
        self._appliances: List[Dict] = []

    # ─────────────────────────────────────────────
    # CRUD Operations
    # ─────────────────────────────────────────────
    def add_appliance(self, appliance_dict: Dict) -> str:
        """
        Adds an appliance to the home profile.

        Args:
            appliance_dict: Dict with keys:
                name, type, quantity, rated_wattage, star_rating,
                inverter_mode, tonnage, setpoint_temp, age_years,
                usage_pattern (optional)

        Returns:
            Generated appliance_id
        """
        app_id = str(uuid.uuid4())[:8]
        appliance_dict["appliance_id"] = app_id

        # Ensure usage_pattern exists
        if "usage_pattern" not in appliance_dict:
            appliance_dict["usage_pattern"] = {day: [] for day in DAYS_OF_WEEK}

        # Compute daily avg hours and monthly estimate
        self._compute_estimates(appliance_dict)

        self._appliances.append(appliance_dict)
        return app_id

    def remove_appliance(self, appliance_id: str) -> bool:
        """Removes an appliance by ID. Returns True if found and removed."""
        before = len(self._appliances)
        self._appliances = [a for a in self._appliances
                            if a.get("appliance_id") != appliance_id]
        return len(self._appliances) < before

    def update_appliance(self, appliance_id: str, updates: Dict) -> bool:
        """Updates an appliance's fields. Recomputes estimates after update."""
        for app in self._appliances:
            if app.get("appliance_id") == appliance_id:
                app.update(updates)
                self._compute_estimates(app)
                return True
        return False

    def get_all_appliances(self) -> List[Dict]:
        """Returns list of all appliances."""
        return self._appliances

    def get_appliance(self, appliance_id: str) -> Optional[Dict]:
        """Returns a single appliance by ID, or None."""
        for app in self._appliances:
            if app.get("appliance_id") == appliance_id:
                return app
        return None

    # ─────────────────────────────────────────────
    # Estimation Methods
    # ─────────────────────────────────────────────
    def _compute_estimates(self, app: Dict) -> None:
        """
        Computes daily_avg_hours, monthly_kwh_estimate, monthly_cost_estimate
        and peak_hours_per_day for an appliance.
        """
        pattern = app.get("usage_pattern", {})
        total_hours_week = 0
        peak_hours_week = 0

        for day in DAYS_OF_WEEK:
            hours = pattern.get(day, [])
            total_hours_week += len(hours)
            peak_hours_week += sum(1 for h in hours if is_peak_hour(h))

        daily_avg = round(total_hours_week / 7, 1)
        quantity  = app.get("quantity", 1)
        wattage   = app.get("rated_wattage", 100)
        age       = app.get("age_years", 0)

        # Age efficiency degradation: ~1.5% per year
        age_factor = 1.0 + (age * 0.015)

        # Monthly kWh = daily_hours × wattage × age_factor × 30 days / 1000
        monthly_kwh = round(daily_avg * (wattage / 1000.0) * age_factor * 30 * quantity, 2)

        # Cost calculation with peak/off-peak split
        peak_daily   = round(peak_hours_week / 7, 1)
        offpeak_daily = max(daily_avg - peak_daily, 0)

        monthly_peak_kwh    = round(peak_daily * (wattage / 1000.0) * age_factor * 30 * quantity, 2)
        monthly_offpeak_kwh = round(offpeak_daily * (wattage / 1000.0) * age_factor * 30 * quantity, 2)

        monthly_cost = round(
            monthly_peak_kwh * PEAK_TARIFF + monthly_offpeak_kwh * OFFPEAK_TARIFF, 2
        )
        peak_extra_cost = round(
            monthly_peak_kwh * (PEAK_TARIFF - OFFPEAK_TARIFF), 2
        )

        app["daily_avg_hours"]       = daily_avg
        app["peak_hours_per_day"]    = peak_daily
        app["monthly_kwh_estimate"]  = monthly_kwh
        app["monthly_cost_estimate"] = monthly_cost
        app["peak_extra_cost"]       = peak_extra_cost

    def calculate_monthly_bill(self) -> Dict:
        """
        Calculates total monthly bill for all appliances.

        Returns:
            Dict with total_kwh, total_cost, per_appliance breakdown.
        """
        total_kwh  = 0.0
        total_cost = 0.0
        breakdown  = []

        for app in self._appliances:
            self._compute_estimates(app)
            kwh  = app.get("monthly_kwh_estimate", 0)
            cost = app.get("monthly_cost_estimate", 0)
            total_kwh  += kwh
            total_cost += cost
            breakdown.append({
                "name":  app.get("name", "Unknown"),
                "type":  app.get("type", "Other"),
                "kwh":   kwh,
                "cost":  cost,
            })

        return {
            "total_kwh":         round(total_kwh, 2),
            "total_cost":        round(total_cost, 2),
            "total_appliances":  len(self._appliances),
            "total_wattage":     sum(a.get("rated_wattage", 0) * a.get("quantity", 1)
                                     for a in self._appliances),
            "per_appliance":     breakdown,
        }

    def get_peak_hour_appliances(self,
                                 peak_hours: Optional[List[int]] = None) -> List[Dict]:
        """
        Returns appliances that are used during peak hours.

        Args:
            peak_hours: List of peak hours (default: 9–22)

        Returns:
            List of appliance dicts that have usage in peak hours.
        """
        peaks = peak_hours or PEAK_HOURS
        result = []
        for app in self._appliances:
            pattern = app.get("usage_pattern", {})
            has_peak = any(
                h in peaks
                for day_hours in pattern.values()
                for h in day_hours
            )
            if has_peak:
                result.append(app)
        return result

    # ─────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────
    def save_to_json(self, filepath: str) -> None:
        """Saves the home profile to a JSON file."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        data = {
            "saved_at":   datetime.now().isoformat(),
            "appliances": self._appliances,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_json(self, filepath: str) -> bool:
        """Loads the home profile from a JSON file. Returns True on success."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._appliances = data.get("appliances", [])
            # Recompute estimates
            for app in self._appliances:
                self._compute_estimates(app)
            return True
        except Exception:
            return False

    def export_csv_string(self) -> str:
        """Exports all appliances as a CSV string."""
        headers = [
            "Name", "Type", "Quantity", "Rated Wattage", "Star Rating",
            "Inverter", "Age Years", "Daily Avg Hours", "Peak Hrs/Day",
            "Monthly kWh", "Monthly Cost ₹", "Peak Extra Cost ₹",
        ]
        lines = [",".join(headers)]
        for app in self._appliances:
            lines.append(",".join(str(x) for x in [
                app.get("name", ""),
                app.get("type", ""),
                app.get("quantity", 1),
                app.get("rated_wattage", 0),
                app.get("star_rating", 3),
                app.get("inverter_mode", False),
                app.get("age_years", 0),
                app.get("daily_avg_hours", 0),
                app.get("peak_hours_per_day", 0),
                app.get("monthly_kwh_estimate", 0),
                app.get("monthly_cost_estimate", 0),
                app.get("peak_extra_cost", 0),
            ]))
        return "\n".join(lines)
