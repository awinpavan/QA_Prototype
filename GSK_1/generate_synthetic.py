import csv
import random
import datetime
import uuid
import os

# --- CONFIGURATION ---

# Ensure the output directory exists
output_dir = "data/synthetic_batches"
os.makedirs(output_dir, exist_ok=True)

# Define the possible anomaly scenarios
ANOMALY_TYPES = [
    "SUDDEN_SPIKE",       # Original sharp deviation
    "GRADUAL_DRIFT",      # Slow deviation over time
    "SENSOR_FAILURE",     # Missing data points
    "CALIBRATION_DUE",    # A note-based anomaly
    "SILENT_SPIKE"        # A spike with no written note
]

def generate_batch_data(batch_id, num_steps, anomaly_type=None):
    """Generates rows for a single batch with multiple metrics and anomaly types."""
    data = []
    fieldnames = ["timestamp", "batch_id", "step", "equipment_id", "metric_name", "metric_value", "unit", "operator", "note"]
    
    # Configuration for each metric, including anomaly parameters
    metrics_config = {
        "temperature": {"unit": "C", "normal_range": (37.0, 39.0), "anomaly_shift": 5.0, "note": "Sudden temperature spike detected"},
        "pressure": {"unit": "PSI", "normal_range": (140.0, 150.0), "anomaly_shift": 15.0, "note": "Pressure rising unexpectedly"},
        "pH": {"unit": "pH", "normal_range": (6.8, 7.2), "anomaly_shift": -0.5, "note": "pH level dropping"},
        "scale_weight": {"unit": "kg", "normal_range": (500.0, 502.0), "anomaly_shift": 0.0, "note": "Calibration overdue for scale S-451"}
    }

    # Simulation state
    equipment_ids = ["reactor-01", "mixer-03", "filler-02"]
    temp_drift = 0.0 # State for gradual drift anomaly

    for i in range(num_steps):
        t = datetime.datetime.now().isoformat()
        step = f"step_{i % 10}"
        operator = random.choice(["op_101", "op_202", "op_303"])
        equipment_id = random.choice(equipment_ids)
        is_anomaly_window = 5 <= (i % 10) <= 8 # Anomaly window for steps 5-8

        for metric_name, config in metrics_config.items():
            metric_value = random.uniform(config["normal_range"][0], config["normal_range"][1])
            note = ""

            if is_anomaly_window and anomaly_type:
                # --- APPLY ANOMALY LOGIC BASED ON TYPE ---
                if anomaly_type in ["SUDDEN_SPIKE", "SILENT_SPIKE"] and metric_name != "scale_weight":
                    metric_value += config["anomaly_shift"]
                    if anomaly_type == "SUDDEN_SPIKE":
                        note = config["note"]
                
                elif anomaly_type == "GRADUAL_DRIFT" and metric_name == "temperature":
                    temp_drift += 0.35 # Increment the drift each step
                    metric_value += temp_drift
                    note = f"Gradual temperature drift observed; current deviation: +{temp_drift:.2f}C"

                elif anomaly_type == "SENSOR_FAILURE" and metric_name == "pressure":
                    if random.random() < 0.6: # 60% chance of failure within the window
                      metric_value = None # Simulate sensor failure
                      note = "Sensor reading unavailable"

                elif anomaly_type == "CALIBRATION_DUE" and metric_name == "scale_weight":
                    note = config["note"] # Add note without changing value

            # Final row assembly
            final_value = round(metric_value, 2) if metric_value is not None else ""
            row = {
                "timestamp": t, "batch_id": batch_id, "step": step,
                "equipment_id": equipment_id, "metric_name": metric_name,
                "metric_value": final_value, "unit": config["unit"],
                "operator": operator, "note": note
            }
            data.append(row)
            
    return data, fieldnames

def write_csv(file_path, data, fieldnames):
    """Writes data to a CSV file."""
    with open(file_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Successfully generated {file_path}")

# --- Main script execution ---
if __name__ == "__main__":
    num_simulation_steps = 100

    # Generate a normal batch for baseline
    batch_normal_id = f"batch_{uuid.uuid4().hex[:6]}"
    batch_normal_data, fieldnames = generate_batch_data(batch_normal_id, num_simulation_steps, anomaly_type=None)
    write_csv(os.path.join(output_dir, "batch_normal_01.csv"), batch_normal_data, fieldnames)
    print("-" * 20)

    # Generate one anomalous batch for each scenario
    for anomaly in ANOMALY_TYPES:
        batch_id = f"batch_{uuid.uuid4().hex[:6]}"
        batch_data, _ = generate_batch_data(batch_id, num_simulation_steps, anomaly_type=anomaly)
        file_name = f"batch_anomaly_{anomaly.lower()}_01.csv"
        write_csv(os.path.join(output_dir, file_name), batch_data, fieldnames)