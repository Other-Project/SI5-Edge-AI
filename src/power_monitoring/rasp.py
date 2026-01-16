import cv2
import google.genai as genai
from PIL import Image
import os
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "power_monitoring"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY=""
client = genai.Client(api_key=API_KEY)
model = 'gemma-3-27b-it' 

def extract_values_to_csv(video_path, output_file):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0: fps = 30 
    
    frame_count = 0
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'voltage_v', 'current_a', 'power_watts'])
        
        print(f"Analysis in progress... Data will be saved in: {output_file}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % int(fps) == 0:
                second = int(frame_count / fps)

                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                if(video_path == "./video_oakd.mp4"):
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_adj = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
                
                pil_image = Image.fromarray(frame_adj)

                prompt = "Analyze the screen. Give ONLY the two numbers separated by a comma (e.g., 5.21, 0.13). Do not add any letters."
                
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=[prompt, pil_image]
                    )
                    clean_res = response.text.strip().replace(' ', '')
                    
                    values = clean_res.split(',')
                    
                    if len(values) == 2:
                        voltage, current = values[0], values[1]
                        writer.writerow([second, voltage, current, float(voltage)*float(current)])
                        print(f"[{second}s] Saved: {voltage}V, {current}A, {float(voltage)*float(current)}W")
                    else:
                        print(f"[{second}s] Format error: {clean_res}")
                        
                except Exception as e:
                    print(f"Error at {second}s: {e}")

            frame_count += 1

    cap.release()
    print(f"Analysis complete. File ready: {os.path.abspath(output_file)}")

extract_values_to_csv("./video_onnx.mp4", str(OUTPUT_DIR / "rasp.csv"))