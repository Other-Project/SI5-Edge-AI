import csv
import time
from pynvml import *
from pathlib import Path

GPU_INDEX = 0 
INTERVAL = 0.5        
OUTPUT_FILE = Path(__file__).resolve().parents[2] / "results" / "power_monitoring" / "nvidia_gpu.csv"

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(GPU_INDEX)

data = []
second = 0

print("Measuring GPU power consumption...")
print("Press CTRL+C to stop and export CSV.")

try:
    while True:
        power_mw = nvmlDeviceGetPowerUsage(handle)
        power_w = power_mw / 1000.0

        data.append([second, power_w])

        print(f"{second}s -> {power_w:.2f}W")
        second += INTERVAL
        time.sleep(INTERVAL)

except KeyboardInterrupt:
    print("\nStop requested, exporting CSV...")

finally:
    nvmlShutdown()

    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "power_watts"])
        writer.writerows(data)

    print(f"Data exported to: {OUTPUT_FILE}")
