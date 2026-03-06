"""
optimization/optimizer.py
5-Rule Energy Optimization Engine.
Analyzes predicted 15-min slots and generates actionable recommendations.
"""

import numpy as np
from typing import Optional
from voltage_calculator import VoltageDropCalculator


class ApplianceOptimizer:
    """
    Analyzes per-slot predictions and applies 5 optimization rules.
    Max total saving capped at 40% of original consumption.
    """

    PEAK_TARIFF    = 8.50
    OFFPEAK_TARIFF = 4.50
    CO2_FACTOR     = 0.82    # kg CO2 per kWh (India grid average)
    TREE_FACTOR    = 21.77   # kg CO2 absorbed per tree per year
    KM_FACTOR      = 0.21    # kg CO2 per km driving

    # Appliances that can be shifted to off-peak hours
    SHIFTABLE = {"Geyser", "Washing Machine", "Electric Iron", "Dishwasher"}

    # Voltage calculator instance
    _voltage_calc = VoltageDropCalculator()

    def optimize_session(
        self,
        session_data:    dict,
        appliance_type:  str,
        setpoint:        Optional[float] = 24.0,
        weather_forecast: Optional[list] = None,
    ) -> dict:
        """
        Main optimization entry point.
        session_data: output of predictor.predict_session()
        Returns full optimization result with per-slot actions + summary.
        """
        slots = session_data.get("slots", [])
        if not slots:
            return self._empty_result()

        optimized_slots = []
        total_saving_kwh = 0.0

        for i, slot in enumerate(slots):
            opt = self._optimize_slot(i, slot, slots, appliance_type, setpoint)
            optimized_slots.append(opt)
            total_saving_kwh += opt["saving_kwh"]

        # Cap at 40% total saving
        original_total = session_data["total_kwh"]
        max_saving = original_total * 0.40
        if total_saving_kwh > max_saving:
            scale = max_saving / total_saving_kwh
            for s in optimized_slots:
                s["saving_kwh"]    = round(s["saving_kwh"] * scale, 5)
                s["saving_inr"]    = round(s["saving_inr"] * scale, 4)
                s["optimized_kwh"] = round(s["original_kwh"] - s["saving_kwh"], 5)
                s["optimized_cost_inr"] = round(s["optimized_kwh"] * s["tariff"], 4)
            total_saving_kwh = sum(s["saving_kwh"] for s in optimized_slots)

        total_orig_kwh   = round(sum(s["original_kwh"]   for s in optimized_slots), 4)
        total_opt_kwh    = round(sum(s["optimized_kwh"]  for s in optimized_slots), 4)
        total_orig_cost  = round(sum(s["original_cost_inr"]  for s in optimized_slots), 4)
        total_opt_cost   = round(sum(s["optimized_cost_inr"] for s in optimized_slots), 4)
        total_saving_inr = round(total_orig_cost - total_opt_cost, 4)
        saving_pct       = round((total_saving_inr / (total_orig_cost + 1e-9)) * 100, 2)

        # Environmental impact
        saved_kwh_val   = round(total_orig_kwh - total_opt_kwh, 4)
        co2_saved       = round(saved_kwh_val * self.CO2_FACTOR, 4)
        trees_equiv     = round(co2_saved / self.TREE_FACTOR, 4)
        km_avoided      = round(co2_saved / self.KM_FACTOR, 2)

        # Top 3 actions by saving
        actions = sorted(
            [s for s in optimized_slots if s["action"]],
            key=lambda x: x["saving_inr"],
            reverse=True,
        )[:3]
        top_3 = [{"time": a["time"], "action": a["action"],
                   "saving_inr": a["saving_inr"], "priority": a["priority"]}
                 for a in actions]

        # Appliance-specific tip
        app_tip = self._appliance_tip(appliance_type)

        return {
            "optimized_slots":       optimized_slots,
            "summary": {
                "total_original_kwh":     total_orig_kwh,
                "total_optimized_kwh":    total_opt_kwh,
                "total_original_cost":    total_orig_cost,
                "total_optimized_cost":   total_opt_cost,
                "total_saving_inr":       total_saving_inr,
                "total_saving_kwh":       round(saved_kwh_val, 4),
                "saving_percent":         saving_pct,
                "monthly_saving_inr":     round(total_saving_inr * 30, 2),
                "yearly_saving_inr":      round(total_saving_inr * 365, 2),
                "co2_saved_kg":           co2_saved,
                "trees_equivalent":       trees_equiv,
                "km_driving_avoided":     km_avoided,
                "top_3_actions":          top_3,
                "appliance_tip":          app_tip,
            },
        }

    # ────────────────────────────────────────
    # PER-SLOT OPTIMIZER
    # ────────────────────────────────────────
    def _optimize_slot(self, i: int, slot: dict, all_slots: list,
                       app: str, setpoint: Optional[float]) -> dict:
        orig_kwh  = slot["kwh"]
        orig_cost = slot["cost_inr"]
        tariff    = slot.get("tariff", self.PEAK_TARIFF)
        temp      = slot.get("outdoor_temp", 32)
        td        = slot.get("thermal_delta", temp - (setpoint or 24))
        is_peak   = slot.get("is_peak", True)

        saving_kwh  = 0.0
        action      = ""
        rule_applied = ""
        priority    = "LOW"

        # ── Rule 1: Setpoint Optimization (AC only) ──────────────
        if app == "AC" and setpoint:
            if td > 18:
                saving_kwh   = orig_kwh * 0.15
                action       = f"Raise AC to {setpoint + 3:.0f}°C — outdoor {temp}°C, delta too high"
                rule_applied = "Rule 1 – Setpoint +3°C"
                priority     = "HIGH"
            elif td > 15:
                saving_kwh   = orig_kwh * 0.10
                action       = f"Raise AC to {setpoint + 2:.0f}°C — outdoor {temp}°C"
                rule_applied = "Rule 1 – Setpoint +2°C"
                priority     = "MEDIUM"

        # ── Rule 2: Peak Hour Shifting ────────────────────────────
        if not action and is_peak and app in self.SHIFTABLE:
            saving_inr_r2 = orig_kwh * (self.PEAK_TARIFF - self.OFFPEAK_TARIFF)
            saving_kwh    = orig_kwh * 0.0    # same energy, lower tariff
            action        = f"Shift {app} to off-peak (after 10 PM) → Save ₹{saving_inr_r2:.2f}"
            rule_applied  = "Rule 2 – Peak Hour Shift"
            priority      = "HIGH"
            # Override cost saving
            saving_inr_override = saving_inr_r2
            opt_kwh  = orig_kwh
            opt_cost = round(orig_kwh * self.OFFPEAK_TARIFF, 4)
            return {
                "time": slot["time"], "original_kwh": orig_kwh,
                "optimized_kwh": opt_kwh,
                "original_cost_inr": orig_cost, "optimized_cost_inr": opt_cost,
                "saving_kwh": 0.0, "saving_inr": round(saving_inr_override, 4),
                "rule_applied": rule_applied, "action": action,
                "priority": priority, "tariff": tariff,
            }

        # ── Rule 3: Eco Mode (temp drop > 3°C) ───────────────────
        if not action and i > 0:
            prev_temp = all_slots[i - 1].get("outdoor_temp", temp)
            if (prev_temp - temp) > 3:
                saving_kwh   = orig_kwh * 0.40
                action       = f"Temp dropped {prev_temp - temp:.1f}°C — Switch to Fan-Only / Eco Mode"
                rule_applied = "Rule 3 – Eco Mode"
                priority     = "HIGH"

        # ── Rule 4: Pre-cooling Strategy ─────────────────────────
        if not action and i + 2 < len(all_slots) and app == "AC":
            next1 = all_slots[i + 1]
            next2 = all_slots[i + 2]
            if next1.get("is_peak") and next2.get("is_peak") and temp > 35:
                saving_kwh   = orig_kwh * 0.15
                action       = f"Pre-cool at 22°C now → raise to {(setpoint or 24) + 2:.0f}°C during peak"
                rule_applied = "Rule 4 – Pre-cooling"
                priority     = "MEDIUM"

        # ── Rule 5: Appliance-specific ───────────────────────────
        # (Applied in summary as a global tip; per-slot action if still none)
        if not action and app == "Refrigerator" and orig_kwh > 0.25:
            saving_kwh   = orig_kwh * 0.10
            action       = "Fridge using high power — check door seals & defrost"
            rule_applied = "Rule 5 – Appliance Tip"
            priority     = "LOW"

        # ── Rule 6: Voltage-Aware Shifting (motor loads during peak) ──
        # Source: IEEE — motor loads draw 11–22% extra current at low voltage
        if not action:
            slot_hour  = slot.get("hour", 0)
            v_data     = self._voltage_calc.calculate_voltage_impact(app, 0, slot_hour)
            grid_v     = v_data["grid_voltage"]
            category   = v_data["category"]
            if category == "motor" and grid_v <= 210:
                v_mult     = v_data["multiplier"]
                extra_pct  = round((v_mult - 1.0) * 100, 1)
                saving_kwh = orig_kwh * (v_mult - 1.0) * 0.80  # 80% recoverable
                action     = (f"⚡ Voltage is {grid_v:.0f}V — {app} drawing +{extra_pct}% extra power. "
                              f"Shift to off-peak (after 11 PM) when voltage is stable at 235V.")
                rule_applied = "Rule 6 – Voltage-Aware Shift"
                priority     = "HIGH"

        saving_kwh  = round(float(np.clip(saving_kwh, 0, orig_kwh * 0.40)), 5)
        opt_kwh     = round(max(orig_kwh - saving_kwh, 0), 5)
        opt_cost    = round(opt_kwh * tariff, 4)
        saving_inr  = round(orig_cost - opt_cost, 4)

        return {
            "time": slot["time"],
            "original_kwh":      orig_kwh,
            "optimized_kwh":     opt_kwh,
            "original_cost_inr": orig_cost,
            "optimized_cost_inr": opt_cost,
            "saving_kwh":        saving_kwh,
            "saving_inr":        max(saving_inr, 0),
            "rule_applied":      rule_applied,
            "action":            action,
            "priority":          priority,
            "tariff":            tariff,
        }

    # ────────────────────────────────────────
    # APPLIANCE-SPECIFIC TIPS
    # ────────────────────────────────────────
    def _appliance_tip(self, app: str) -> str:
        tips = {
            "AC":              "🧹 Clean AC filters monthly — saves 10–15% energy. ⚡ Run AC after 11 PM when voltage is 235V vs 205V during evening peak — saves up to 22% extra power draw.",
            "Geyser":          "⏰ Heat water at 6 AM (off-peak) — save ₹14/day. ⚡ Geyser runs 26% longer at 205V evening voltage.",
            "Refrigerator":    "🚪 Don't open fridge frequently — saves 10%. ⚡ Motor draws +22% current during evening peak voltage drop.",
            "Washing Machine": "🧺 Use cold water + run full load on weekends. ⚡ Motor loads are most affected by voltage drops — run after 11 PM.",
            "Microwave":       "✅ Microwave is efficient. ⚡ Resistive load — runs slightly longer at low voltage.",
            "Ceiling Fan":     "💡 Replace with BLDC fan — saves 65% power. ⚡ Motor fan draws extra at low voltage — BLDC is immune.",
            "LED TV":          "🌑 Enable auto-brightness — saves 20%. ✅ SMPS electronics: voltage-immune.",
            "Desktop PC":      "💤 Enable sleep mode when idle — saves 80%. ✅ SMPS: voltage-stable.",
            "Electric Iron":   "👔 Iron all clothes in one session. ⚡ Iron runs 26% longer at 205V to reach same heat.",
            "LED Bulb":        "✅ LED bulbs are efficient. ✅ SMPS electronics: voltage-immune.",
            "Laptop":          "🔋 Keep battery at 20–80% for longevity. ✅ SMPS: voltage-stable.",
            "Wi-Fi Router":    "📶 Schedule auto-off at night — saves ₹4/month. ✅ SMPS: voltage-stable.",
        }
        return tips.get(app, "✅ No specific tip — appliance is operating efficiently")

    def _empty_result(self) -> dict:
        return {"optimized_slots": [], "summary": {}}
