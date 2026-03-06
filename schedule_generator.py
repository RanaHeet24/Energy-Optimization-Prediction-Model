"""
schedule_generator.py
24-Hour Optimized Energy Schedule Generator.

Auto-detects peak hours and generates a complete optimized daily schedule
with before-vs-after comparison for every 15-minute slot. Uses the home
profile's appliance data and usage patterns to apply 5 optimization rules.

Tariff Tiers (India residential grid):
  Tier 1 — CRITICAL PEAK:  18:00–23:00  ₹8.50/kWh  ~205V
  Tier 2 — MORNING PEAK:   06:00–10:00  ₹7.50/kWh  ~215V
  Tier 3 — OFF-PEAK DAY:   10:00–18:00  ₹6.00/kWh  ~225V
  Tier 4 — CHEAPEST NIGHT: 23:00–06:00  ₹4.50/kWh  ~235V
"""

from typing import Dict, List, Optional
import copy


# ── Tariff Tier Definitions ──────────────────────────────
TARIFF_TIERS = {
    "CRITICAL_PEAK": {"hours": list(range(18, 23)), "rate": 8.50,
                      "voltage": 205, "motor_mult": 1.22, "color": "🔴"},
    "MORNING_PEAK":  {"hours": list(range(6, 10)),  "rate": 7.50,
                      "voltage": 215, "motor_mult": 1.11, "color": "🟡"},
    "OFFPEAK_DAY":   {"hours": list(range(10, 18)), "rate": 6.00,
                      "voltage": 225, "motor_mult": 1.04, "color": "🟠"},
    "CHEAPEST":      {"hours": list(range(23, 24)) + list(range(0, 6)),
                      "rate": 4.50, "voltage": 235, "motor_mult": 1.00,
                      "color": "🟢"},
}

# ── Appliance Categories ────────────────────────────────
SHIFTABLE  = {"Washing Machine", "Dishwasher", "Electric Iron", "Geyser"}
MOTOR_LOADS = {"AC", "Refrigerator", "Washing Machine", "Ceiling Fan"}
NON_SHIFTABLE = {"Refrigerator", "LED Bulb", "Wi-Fi Router", "Laptop",
                 "Desktop PC", "LED TV", "Ceiling Fan"}

# ── Default wattages ────────────────────────────────────
DEFAULT_WATTS = {
    "AC": 1500, "Refrigerator": 250, "Geyser": 2000,
    "Washing Machine": 2000, "Microwave": 1200, "Ceiling Fan": 75,
    "LED TV": 150, "Desktop PC": 350, "Electric Iron": 1200,
    "LED Bulb": 12, "Laptop": 65, "Wi-Fi Router": 8,
    "Dishwasher": 1800, "Air Purifier": 55,
}


def _get_tier_for_hour(hour: int) -> dict:
    """Returns the tariff tier dict for a given hour."""
    for name, tier in TARIFF_TIERS.items():
        if hour in tier["hours"]:
            return {"name": name, **tier}
    return {"name": "CHEAPEST", **TARIFF_TIERS["CHEAPEST"]}


def _get_tier_name(hour: int) -> str:
    return _get_tier_for_hour(hour)["name"]


class ScheduleGenerator:
    """
    Generates, optimizes, and compares 24-hour appliance schedules.

    Usage:
        gen = ScheduleGenerator()
        normal = gen.generate_normal_schedule(home_profile_appliances)
        optimized = gen.generate_optimized_schedule(home_profile_appliances)
        comparison = gen.compare_schedules(normal, optimized)
        summary = gen.generate_summary(comparison)
    """

    # ═════════════════════════════════════════════
    # GENERATE NORMAL SCHEDULE
    # ═════════════════════════════════════════════
    def generate_normal_schedule(
        self,
        appliances: List[Dict],
        weather_data: Optional[Dict] = None,
    ) -> List[Dict]:
        """
        Generates the "normal" unoptimized 24-hour schedule based on
        the user's current appliance usage patterns.

        Creates 96 slots (24 hours × 4 per hour). For each slot,
        identifies which appliances are active and calculates cost
        using the appropriate tariff tier and voltage multiplier.

        Args:
            appliances: List of appliance dicts from HomeProfile
            weather_data: Optional weather dict (unused for now)

        Returns:
            List of 96 slot dicts with energy/cost data.
        """
        slots = []
        cumulative_cost = 0.0

        for slot_idx in range(96):
            hour = slot_idx // 4
            minute = (slot_idx % 4) * 15
            time_str = f"{hour:02d}:{minute:02d}"

            tier = _get_tier_for_hour(hour)
            rate = tier["rate"]
            voltage = tier["voltage"]
            motor_mult = tier["motor_mult"]

            active_apps = []
            total_watts = 0.0
            total_kwh = 0.0
            voltage_extra = 0.0

            for app in appliances:
                app_type = app.get("type", "Other")
                pattern = app.get("usage_pattern", {})
                quantity = app.get("quantity", 1)
                wattage = app.get("rated_wattage",
                                  DEFAULT_WATTS.get(app_type, 100))
                age = app.get("age_years", 0)
                age_factor = 1.0 + age * 0.015

                # Check if this appliance is active at this hour
                # (any day pattern containing this hour counts)
                any_day_active = any(
                    hour in pattern.get(day, [])
                    for day in pattern
                )
                if not any_day_active:
                    continue

                # Apply voltage multiplier for motor loads
                mult = motor_mult if app_type in MOTOR_LOADS else 1.0
                effective_watts = wattage * mult * age_factor * quantity
                slot_kwh = effective_watts * 0.25 / 1000  # 15-min slot

                # Voltage extra cost
                base_kwh = wattage * age_factor * quantity * 0.25 / 1000
                v_extra_kwh = slot_kwh - base_kwh
                v_extra_cost = v_extra_kwh * rate

                active_apps.append(app.get("name", app_type))
                total_watts += effective_watts
                total_kwh += slot_kwh
                voltage_extra += v_extra_cost

            cost = round(total_kwh * rate, 4)
            cumulative_cost += cost

            slots.append({
                "slot_index":          slot_idx,
                "time":                time_str,
                "hour":                hour,
                "active_appliances":   active_apps,
                "total_kwh":           round(total_kwh, 5),
                "total_watts":         round(total_watts, 1),
                "tariff_tier":         tier["name"],
                "rate_per_kwh":        rate,
                "cost_inr":            cost,
                "grid_voltage":        voltage,
                "voltage_extra_cost":  round(voltage_extra, 4),
                "cumulative_cost":     round(cumulative_cost, 2),
            })

        return slots

    # ═════════════════════════════════════════════
    # GENERATE OPTIMIZED SCHEDULE
    # ═════════════════════════════════════════════
    def generate_optimized_schedule(
        self,
        appliances: List[Dict],
        weather_data: Optional[Dict] = None,
        precooling: bool = True,
        shift_loads: bool = True,
        voltage_opt: bool = True,
    ) -> List[Dict]:
        """
        Generates an optimized 24-hour schedule by applying 5 rules:
          Rule 1: Shift shiftable loads to off-peak
          Rule 2: Pre-cooling for AC
          Rule 3: Stagger heavy loads
          Rule 4: Geyser timing (move to 5 AM)
          Rule 5: AC setpoint schedule

        Args:
            appliances:  List of appliance dicts from HomeProfile
            precooling:  Enable pre-cooling strategy
            shift_loads: Enable load shifting
            voltage_opt: Enable voltage optimization

        Returns:
            List of 96 slot dicts with optimization info.
        """
        # Build a mutable per-hour activation map
        # hour -> list of (app_name, app_type, wattage, quantity, age, ...)
        hour_map = {}
        for h in range(24):
            hour_map[h] = []

        optimizations = []  # track actions for action cards

        for app in appliances:
            app_name = app.get("name", app.get("type", "Other"))
            app_type = app.get("type", "Other")
            wattage  = app.get("rated_wattage",
                               DEFAULT_WATTS.get(app_type, 100))
            quantity = app.get("quantity", 1)
            age      = app.get("age_years", 0)
            pattern  = app.get("usage_pattern", {})
            inverter = app.get("inverter_mode", False)

            active_hours = sorted(set(
                h for day_hours in pattern.values() for h in day_hours
            ))

            optimized_hours = list(active_hours)
            applied_rule = None
            reason = None

            # ── Rule 1: Shift shiftable loads to off-peak ──
            if shift_loads and app_type in SHIFTABLE:
                peak_hours_used = [
                    h for h in active_hours
                    if _get_tier_name(h) in ("CRITICAL_PEAK", "MORNING_PEAK")
                ]
                if peak_hours_used:
                    # Find cheapest available slots
                    cheapest = TARIFF_TIERS["CHEAPEST"]["hours"]
                    duration = len(active_hours)
                    new_hours = cheapest[:duration] if duration <= len(cheapest) \
                        else cheapest + TARIFF_TIERS["OFFPEAK_DAY"]["hours"][
                            :duration - len(cheapest)]
                    optimized_hours = sorted(new_hours[:duration])

                    old_tier = _get_tier_for_hour(peak_hours_used[0])
                    new_tier = _get_tier_for_hour(optimized_hours[0])
                    saving = round(
                        (wattage * quantity * len(peak_hours_used) / 1000)
                        * (old_tier["rate"] - new_tier["rate"]), 2
                    )
                    applied_rule = "Rule 1 – Load Shift"
                    reason = (f"Shifted {app_name} from peak ({old_tier['rate']}/kWh)"
                              f" to off-peak ({new_tier['rate']}/kWh)")
                    optimizations.append({
                        "time": f"{optimized_hours[0]:02d}:00",
                        "hour": optimized_hours[0],
                        "appliance": app_name,
                        "rule": applied_rule,
                        "reason": reason,
                        "saving_per_day": saving,
                        "icon": "🔄",
                    })

            # ── Rule 4: Geyser timing to 5 AM ──
            if shift_loads and app_type == "Geyser":
                morning_peak = [h for h in active_hours if 6 <= h <= 9]
                if morning_peak:
                    # Move to 5:00–5:45 AM
                    optimized_hours = [h for h in optimized_hours
                                       if h not in morning_peak]
                    optimized_hours = sorted(set(optimized_hours + [5]))
                    saving = round(
                        (wattage * quantity * len(morning_peak) / 1000)
                        * (7.50 - 4.50), 2
                    )
                    applied_rule = "Rule 4 – Geyser Timing"
                    reason = ("Moved Geyser to 5:00 AM (₹4.50/kWh night rate) "
                              "— hot water stays 3+ hrs in insulated tank")
                    optimizations.append({
                        "time": "05:00",
                        "hour": 5,
                        "appliance": app_name,
                        "rule": applied_rule,
                        "reason": reason,
                        "saving_per_day": saving,
                        "icon": "🚿",
                    })

            # ── Rule 2: Pre-cooling for AC ──
            if precooling and app_type == "AC":
                evening_peak = [h for h in active_hours if 18 <= h <= 22]
                if evening_peak:
                    precool_hour = min(evening_peak) - 2
                    if precool_hour >= 0:
                        # Add pre-cool hour, keep evening as maintenance only
                        if precool_hour not in optimized_hours:
                            optimized_hours = sorted(
                                set(optimized_hours + [precool_hour]))
                        # Saving = reduced consumption during peak
                        saving = round(
                            (wattage * quantity * len(evening_peak) / 1000)
                            * 0.30 * 8.50, 2  # 30% reduction during peak
                        )
                        applied_rule = "Rule 2 – Pre-Cooling"
                        reason = (f"Pre-cool at {precool_hour}:00 (26°C) — "
                                  f"AC runs maintenance-only during 6–11 PM peak")
                        optimizations.append({
                            "time": f"{precool_hour:02d}:00",
                            "hour": precool_hour,
                            "appliance": app_name,
                            "rule": applied_rule,
                            "reason": reason,
                            "saving_per_day": saving,
                            "icon": "❄️",
                        })

            # Populate hour map
            for h in optimized_hours:
                entry = {
                    "name":      app_name,
                    "type":      app_type,
                    "wattage":   wattage,
                    "quantity":  quantity,
                    "age":       age,
                    "rule":      applied_rule,
                    "reason":    reason,
                    "inverter":  inverter,
                }
                # ── Rule 5: AC setpoint schedule ──
                if app_type == "AC":
                    if 18 <= h <= 22:
                        entry["setpoint"] = 26 if precooling else 25
                        entry["ac_cap_frac"] = 0.40 if precooling else 0.70
                        entry["rule"] = entry["rule"] or "Rule 5 – Setpoint"
                    elif 23 <= h or h <= 5:
                        entry["setpoint"] = 28
                        entry["ac_cap_frac"] = 0.30
                    else:
                        entry["setpoint"] = 26
                        entry["ac_cap_frac"] = 0.50
                else:
                    entry["ac_cap_frac"] = 1.0

                hour_map[h].append(entry)

        # ── Rule 3: Stagger heavy loads ──
        HEAVY = {"Geyser", "Washing Machine", "AC", "Dishwasher",
                 "Electric Iron"}
        for h in range(24):
            heavy_in_slot = [e for e in hour_map[h] if e["type"] in HEAVY]
            if len(heavy_in_slot) > 2:
                # Stagger: move excess to next hour
                for excess in heavy_in_slot[2:]:
                    next_h = (h + 1) % 24
                    hour_map[h].remove(excess)
                    excess_copy = dict(excess)
                    excess_copy["rule"] = "Rule 3 – Stagger"
                    excess_copy["reason"] = (
                        f"Staggered {excess['name']} by 30 min to reduce "
                        f"simultaneous heavy-load demand")
                    hour_map[next_h].append(excess_copy)
                    optimizations.append({
                        "time": f"{next_h:02d}:00",
                        "hour": next_h,
                        "appliance": excess["name"],
                        "rule": "Rule 3 – Stagger",
                        "reason": excess_copy["reason"],
                        "saving_per_day": 0,
                        "icon": "⏱️",
                    })

        # Build slot output
        slots = []
        cumulative_cost = 0.0

        for slot_idx in range(96):
            hour = slot_idx // 4
            minute = (slot_idx % 4) * 15
            time_str = f"{hour:02d}:{minute:02d}"

            tier = _get_tier_for_hour(hour)
            rate = tier["rate"]
            voltage = tier["voltage"]
            motor_mult = tier["motor_mult"]

            active_apps = []
            total_watts = 0.0
            total_kwh = 0.0
            voltage_extra = 0.0
            slot_rule = None
            slot_reason = None

            for entry in hour_map.get(hour, []):
                app_type = entry["type"]
                wattage  = entry["wattage"]
                quantity = entry["quantity"]
                age_f    = 1.0 + entry["age"] * 0.015
                cap_frac = entry.get("ac_cap_frac", 1.0)

                # Voltage optimization: if enabled, use off-peak mult
                mult = 1.0
                if voltage_opt and app_type in MOTOR_LOADS:
                    mult = motor_mult
                elif app_type in MOTOR_LOADS:
                    mult = motor_mult

                effective_w = wattage * mult * age_f * quantity * cap_frac
                slot_kwh = effective_w * 0.25 / 1000

                base_kwh = wattage * age_f * quantity * cap_frac * 0.25 / 1000
                v_extra = (slot_kwh - base_kwh) * rate

                active_apps.append(entry["name"])
                total_watts += effective_w
                total_kwh += slot_kwh
                voltage_extra += v_extra

                if entry.get("rule"):
                    slot_rule = entry["rule"]
                    slot_reason = entry.get("reason", "")

            cost = round(total_kwh * rate, 4)
            cumulative_cost += cost

            slots.append({
                "slot_index":            slot_idx,
                "time":                  time_str,
                "hour":                  hour,
                "active_appliances":     active_apps,
                "total_kwh":             round(total_kwh, 5),
                "total_watts":           round(total_watts, 1),
                "tariff_tier":           tier["name"],
                "rate_per_kwh":          rate,
                "cost_inr":              cost,
                "grid_voltage":          voltage,
                "voltage_extra_cost":    round(voltage_extra, 4),
                "cumulative_cost":       round(cumulative_cost, 2),
                "optimization_applied":  slot_rule,
                "reason":                slot_reason,
            })

        # Attach optimizations list for action cards
        self._last_optimizations = sorted(optimizations, key=lambda x: x["hour"])
        return slots

    def get_action_cards(self) -> List[Dict]:
        """Returns the list of optimization actions from last schedule."""
        return getattr(self, "_last_optimizations", [])

    # ═════════════════════════════════════════════
    # COMPARE SCHEDULES
    # ═════════════════════════════════════════════
    def compare_schedules(
        self, normal: List[Dict], optimized: List[Dict]
    ) -> List[Dict]:
        """
        Compares normal vs optimized schedules slot by slot.

        Returns 96 comparison dicts with savings per slot.
        """
        comparison = []
        for n, o in zip(normal, optimized):
            kwh_saved  = round(n["total_kwh"] - o["total_kwh"], 5)
            cost_saved = round(n["cost_inr"] - o["cost_inr"], 4)
            comparison.append({
                "time":                       n["time"],
                "hour":                       n["hour"],
                "tariff_tier":                n["tariff_tier"],
                "rate_per_kwh":               n["rate_per_kwh"],
                "normal_kwh":                 n["total_kwh"],
                "optimized_kwh":              o["total_kwh"],
                "normal_cost":                n["cost_inr"],
                "optimized_cost":             o["cost_inr"],
                "kwh_saved":                  kwh_saved,
                "cost_saved":                 cost_saved,
                "optimization_applied":       o.get("optimization_applied"),
                "reason":                     o.get("reason"),
                "active_appliances_normal":   n["active_appliances"],
                "active_appliances_optimized": o["active_appliances"],
                "normal_cumulative":          n["cumulative_cost"],
                "optimized_cumulative":       o["cumulative_cost"],
            })
        return comparison

    # ═════════════════════════════════════════════
    # GENERATE SUMMARY
    # ═════════════════════════════════════════════
    def generate_summary(self, comparison: List[Dict]) -> Dict:
        """
        Generates aggregate summary from comparison data.

        Returns:
            Dict with totals, savings breakdown, projections, and
            environmental impact.
        """
        total_n_kwh = round(sum(c["normal_kwh"] for c in comparison), 4)
        total_o_kwh = round(sum(c["optimized_kwh"] for c in comparison), 4)
        total_n_cost = round(sum(c["normal_cost"] for c in comparison), 2)
        total_o_cost = round(sum(c["optimized_cost"] for c in comparison), 2)

        kwh_saved  = round(total_n_kwh - total_o_kwh, 4)
        cost_saved = round(total_n_cost - total_o_cost, 2)
        saving_pct = round((cost_saved / (total_n_cost + 1e-9)) * 100, 1)

        # Savings by tier
        peak_saving = round(sum(
            c["cost_saved"] for c in comparison
            if c["tariff_tier"] == "CRITICAL_PEAK"
        ), 2)
        offpeak_saving = round(sum(
            c["cost_saved"] for c in comparison
            if c["tariff_tier"] == "CHEAPEST"
        ), 2)

        # Savings by strategy (from action cards)
        actions = self.get_action_cards()
        precool_s = sum(a["saving_per_day"] for a in actions
                        if "Pre-Cool" in a.get("rule", ""))
        shift_s   = sum(a["saving_per_day"] for a in actions
                        if "Load Shift" in a.get("rule", ""))
        geyser_s  = sum(a["saving_per_day"] for a in actions
                        if "Geyser" in a.get("rule", ""))
        stagger_s = sum(a["saving_per_day"] for a in actions
                        if "Stagger" in a.get("rule", ""))

        # Voltage saving = difference not covered by above
        voltage_s = max(0, cost_saved - precool_s - shift_s - geyser_s - stagger_s)

        # Best/worst hours
        if comparison:
            best_slot = max(comparison, key=lambda c: c["cost_saved"])
            worst_slot = max(comparison, key=lambda c: c["normal_cost"])
        else:
            best_slot = {"time": "N/A"}
            worst_slot = {"time": "N/A"}

        # Environmental impact
        co2_saved = round(kwh_saved * 0.82, 2)    # India emission factor
        trees     = round(co2_saved * 30 / 21.77, 1)  # 1 tree absorbs ~21.77 kg CO2/yr

        return {
            "total_normal_kwh":     total_n_kwh,
            "total_optimized_kwh":  total_o_kwh,
            "total_kwh_saved":      kwh_saved,
            "total_normal_cost":    total_n_cost,
            "total_optimized_cost": total_o_cost,
            "total_cost_saved":     cost_saved,
            "saving_percent":       saving_pct,
            "peak_hour_saving":     peak_saving,
            "off_peak_saving":      offpeak_saving,
            "voltage_saving":       round(voltage_s, 2),
            "precooling_saving":    round(precool_s, 2),
            "load_shifting_saving": round(shift_s, 2),
            "geyser_timing_saving": round(geyser_s, 2),
            "stagger_saving":       round(stagger_s, 2),
            "monthly_saving":       round(cost_saved * 30, 2),
            "yearly_saving":        round(cost_saved * 365, 2),
            "co2_saved_kg":         co2_saved,
            "trees_equivalent":     trees,
            "best_saving_hour":     best_slot["time"],
            "worst_usage_hour":     worst_slot["time"],
        }
