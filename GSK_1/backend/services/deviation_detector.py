import csv
import os
from typing import List, Dict, Any

# --- Configuration: Define Deviation Rules ---

# Rule 1: Normal operating thresholds for numeric values
METRIC_THRESHOLDS = {
    "temperature": {"min": 35.0, "max": 40.0},
    "pressure": {"min": 135.0, "max": 155.0},
    "pH": {"min": 6.7, "max": 7.3},
    "scale_weight": {"min": 500.0, "max": 502.0}
}

# Rule 2: Keywords in the 'note' column that indicate a deviation
NOTE_KEYWORDS = ["calibration", "overdue", "failure", "unavailable", "drift", "spike"]


def find_deviations_in_file(batch_csv_path: str) -> List[Dict[str, Any]]:
    """
    Reads a single batch CSV file and identifies any rows where metric values
    fall outside the predefined numeric thresholds, are null, or contain flagged keywords.

    Args:
        batch_csv_path: The full path to the batch data CSV file.

    Returns:
        A list of dictionaries, where each dictionary is a row that was
        flagged as a deviation.
    """
    deviations = []
    try:
        with open(batch_csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric_name = row.get("metric_name")
                metric_value_str = row.get("metric_value", "").strip()
                note_text = row.get("note", "").lower()

                # Check 1: Null/empty values (Sensor Failure)
                if not metric_value_str:
                    row['deviation_reason'] = f"Missing value for '{metric_name}' (Sensor Failure)"
                    deviations.append(row)
                    continue # Prioritize this reason and move to the next row

                # Check 2: Numeric value out of range
                is_out_of_range = False
                if metric_name in METRIC_THRESHOLDS:
                    try:
                        metric_value = float(metric_value_str)
                        thresholds = METRIC_THRESHOLDS[metric_name]
                        min_val, max_val = thresholds["min"], thresholds["max"]
                        
                        if not (min_val <= metric_value <= max_val):
                            row['deviation_reason'] = f"Value '{metric_value}' is outside range ({min_val}-{max_val})"
                            deviations.append(row)
                            is_out_of_range = True
                            
                    except (ValueError, TypeError):
                        pass # Ignore non-numeric values for this check
                
                if is_out_of_range:
                    continue # Prioritize value deviation reason, move to next row

                # Check 3: Flagged keywords in the 'note' column
                for keyword in NOTE_KEYWORDS:
                    if keyword in note_text:
                        row['deviation_reason'] = f"Note contains flagged keyword: '{keyword}'"
                        deviations.append(row)
                        break # Found a keyword, no need to check for others
                        
    except FileNotFoundError:
        print(f"Error: The file '{batch_csv_path}' was not found.")
        return []
        
    return deviations

def find_all_deviations_in_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    Scans a directory for .csv files and finds all deviations across them.

    Args:
        directory_path: The path to the directory containing batch CSV files.

    Returns:
        A single list containing all deviation rows found in all CSV files.
    """
    all_deviations = []
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        return []
    
    print(f"--- Scanning directory: {directory_path} ---")
    for filename in sorted(os.listdir(directory_path)):
        if filename.endswith(".csv"):
            file_path = os.path.join(directory_path, filename)
            print(f"  - Analyzing {filename}...")
            file_deviations = find_deviations_in_file(file_path)
            if file_deviations:
                # Add source file context to each deviation row
                for dev in file_deviations:
                    dev['source_file'] = filename
                all_deviations.extend(file_deviations)
    return all_deviations


# --- Example Usage (for testing the script directly) ---
if __name__ == "__main__":
    batches_directory = "data/synthetic_batches"
    all_found_deviations = find_all_deviations_in_directory(batches_directory)
    
    print("\n" + "="*50 + "\n")
    
    if all_found_deviations:
        print(f"Found a total of {len(all_found_deviations)} deviations across all files.")
        
        # Group deviations by file for the classic, clearer output
        deviations_by_file = {}
        for dev in all_found_deviations:
            source = dev['source_file']
            if source not in deviations_by_file:
                deviations_by_file[source] = []
            deviations_by_file[source].append(dev)

        for file, devs in deviations_by_file.items():
             print(f"\nDeviations in '{file}' ({len(devs)} found):")
             # Print the first 5 deviations for a quick look
             for dev in devs[:5]:
                 print(f"  - Reason: {dev.get('deviation_reason', 'N/A')}")
                 print(f"    L Timestamp: {dev['timestamp']}, Metric: {dev['metric_name']}, Value: '{dev['metric_value']}' {dev['unit']}")
             if len(devs) > 5:
                 print(f"  ... and {len(devs) - 5} more.")
    else:
        print("No deviations found in any batch files.")

