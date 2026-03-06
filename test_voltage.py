"""Quick smoke test for voltage drop integration."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voltage_calculator import VoltageDropCalculator
from predictor import predict_session

print("=== Voltage Calculator Tests ===")
vc = VoltageDropCalculator()

# Test 1: Grid voltage by hour
for h in [3, 8, 14, 20]:
    v = vc.get_grid_voltage(h)
    print(f"  Hour {h:2d}: {v}V")

# Test 2: Appliance categories
for app in ["AC", "Geyser", "LED TV"]:
    cat = vc.get_appliance_type_category(app)
    print(f"  {app}: {cat}")

# Test 3: Full impact for AC at evening peak
r = vc.calculate_voltage_impact("AC", 1500, 20)
print(f"\n  AC 1500W at 8PM (evening peak):")
print(f"    Grid Voltage: {r['grid_voltage']}V")
print(f"    Multiplier:   {r['multiplier']}")
print(f"    Actual Watts: {r['actual_watts']}W")
print(f"    Extra Watts:  +{r['extra_watts']}W")
print(f"    Warning:      {r['warning']}")

# Test 4: Predict session with voltage
print("\n=== Predictor Integration Test ===")
weather = {"outdoor_temp": 35.0, "humidity": 55.0}
result = predict_session(
    appliance_type="AC", rated_wattage=1500, star_rating=3,
    inverter_mode=1, setpoint_temp=24.0, start_hour=20,
    start_minute=0, duration_hours=1.0, weather_data=weather,
    occupancy=3,
)
print(f"  Total kWh:          {result['total_kwh']}")
print(f"  Voltage Extra kWh:  {result['voltage_extra_kwh']}")
print(f"  Voltage Extra Cost: Rs.{result['voltage_extra_cost']}")
print(f"  Slots with voltage: {len([s for s in result['slots'] if 'voltage' in s])}/{len(result['slots'])}")

slot0 = result["slots"][0]
print(f"  Slot 1: kwh_base={slot0['kwh_base']}, kwh={slot0['kwh']}, voltage={slot0['voltage']}V, mult={slot0['voltage_multiplier']}")

print("\n=== ALL TESTS PASSED ===")
