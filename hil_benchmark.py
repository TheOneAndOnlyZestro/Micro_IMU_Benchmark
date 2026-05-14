import serial
import time
import struct
import os
import csv
import pandas as pd
import numpy as np
import random  # <--- Added for shuffling

# --- CONFIGURATION ---
SERIAL_PORT = '/dev/tty.usbserial-0001'  
BAUD_RATE = 115200
MODELS_DIR = 'tflite_models'
DATA_FILE = 'dataset/IMU_Data_5.csv'
RESULTS_FILE = 'benchmark_results'

FEATURE_COUNT = 9
OUTPUT_COUNT = 7

def get_seq_len_from_filename(filename):
    name_without_ext = filename.replace('.tflite', '')
    parts = name_without_ext.split('_')
    for anchor in ['True', 'False']:
        if anchor in parts:
            idx = parts.index(anchor)
            try: return int(parts[idx - 1])
            except (ValueError, IndexError): pass
    try: return int(parts[-3])
    except Exception: return None

def wait_for_response(ser, prefixes, timeout=8.0):
    if isinstance(prefixes, str): prefixes = [prefixes]
    start = time.time()
    while time.time() - start < timeout:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if any(p in line for p in prefixes):
                return line
            else:
                print(f"    [ESP32 LOG]: {line}")
    return None

def send_model(ser, model_path):
    print(f"--> Uploading Model: {os.path.basename(model_path)}")
    with open(model_path, "rb") as f:
        model_data = f.read()
    
    ser.write(f"M{len(model_data)}\n".encode())
    
    res = wait_for_response(ser, ["ACK_SIZE", "ERR_SIZE"], timeout=5.0)
    if not res:
        return "ERROR_TIMEOUT_SIZE"
    if res.startswith("ERR_SIZE"):
        return f"ERROR_ESP32_MALLOC_FAILED, {res}"

    chunk_size = 4096
    for i in range(0, len(model_data), chunk_size):
        ser.write(model_data[i:i+chunk_size])
        ack = wait_for_response(ser, ["CHUNK_OK", "ERR_FLASH_WRITE"], timeout=2.0)
        if ack is None or "ERR" in ack:
            return f"ERROR_CHUNK_FAILED, {ack}"
        #time.sleep(0.02)

    res = wait_for_response(ser, ["ACK_M", "ERR_INIT"], timeout=10.0)
    if res and res.startswith("ACK_M"):
        return "SUCCESS"
    else:
        return "ERROR_TFLITE_INIT_FAILED"

def run_inference(ser, feature_array):
    flat_data = feature_array.flatten().astype(np.float32)
    binary_data = struct.pack(f'<{len(flat_data)}f', *flat_data)
    
    ser.write(f"D{len(flat_data)}\n".encode())
    ser.write(binary_data)
    
    res = wait_for_response(ser, ["RES,", "ERR_"], timeout=10.0)
    if res and res.startswith("RES,"):
        parts = res.split(',')
        time_us = int(parts[1])
        arena_bytes = int(parts[2])
        predictions = [float(x) for x in parts[3:]]
        return time_us, arena_bytes, predictions
    elif res:
        return res, None, None 
    return "ERROR_TIMEOUT", None, None

def append_to_csv(filepath, record):
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=record.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)

def main():
    print("Initializing PC-ESP32 HIL Benchmark System...")
    
    os.makedirs('benchmarks', exist_ok=True)
    csv_filename = os.path.join('benchmarks', f"{RESULTS_FILE}_{int(time.time())}.csv")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        ser.reset_input_buffer()
        print(f"Connected to ESP32 on {SERIAL_PORT}")

        print("--- ESP32 boot output ---")
        deadline = time.time() + 2.0
        while time.time() < deadline:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').rstrip()
                print(f"  {line!r}")
        print("--- end boot output ---")
    except Exception as e:
        print(f"Failed to open Serial: {e}")
        return

    df = pd.read_csv(DATA_FILE)
    features = df.iloc[:, :FEATURE_COUNT].values
    labels = df.iloc[:, FEATURE_COUNT:FEATURE_COUNT+OUTPUT_COUNT].values
    
    tflite_files = [f for f in os.listdir(MODELS_DIR)]
    
    # ========================================================
    # NEW: SHUFFLE THE MODELS TO PREVENT CASCADING ERROR BIAS
    # ========================================================
    random.shuffle(tflite_files)
    print(f"Randomized order of {len(tflite_files)} models for testing.")

    for tflite_file in tflite_files:
        seq_len = get_seq_len_from_filename(tflite_file)
        if seq_len is None: continue
            
        print(f"\n======================================")
        print(f"Model: {tflite_file} | Extracted Seq Len: {seq_len}")
        print(f"======================================")

        test_sequences, test_labels = [], []
        for i in range(5):
            if i + seq_len <= len(features):
                test_sequences.append(features[i:i+seq_len])
                test_labels.append(labels[i+seq_len-1])
        
        model_path = os.path.join(MODELS_DIR, tflite_file)
        upload_status = send_model(ser, model_path)
        
        if upload_status != "SUCCESS":
            print(f"    [Error] {upload_status}")
            record = {
                'Model': tflite_file, 'Seq_Length': seq_len, 'Sample_ID': 'N/A',
                'Inference_Time_ms': upload_status, 'Arena_Memory_Bytes': upload_status, 'MSE': upload_status
            }
            append_to_csv(csv_filename, record)
            continue

        for idx, (seq, true_label) in enumerate(zip(test_sequences, test_labels)):
            time_us_or_err, arena, preds = run_inference(ser, seq)
            if preds is not None:
                mse = np.mean((np.array(preds) - true_label) ** 2)
                time_ms = time_us_or_err / 1000.0
                print(f"    Sample {idx}: Time={time_ms:.1f}ms | Memory={arena}B | MSE={mse:.4f}")
                record = {
                    'Model': tflite_file, 'Seq_Length': seq_len, 'Sample_ID': idx,
                    'Inference_Time_ms': time_ms, 'Arena_Memory_Bytes': arena, 'MSE': mse
                }
            else:
                print(f"    Sample {idx}: Failed ({time_us_or_err})")
                record = {
                    'Model': tflite_file, 'Seq_Length': seq_len, 'Sample_ID': idx,
                    'Inference_Time_ms': time_us_or_err, 'Arena_Memory_Bytes': 'ERROR', 'MSE': 'ERROR'
                }
            append_to_csv(csv_filename, record)

    ser.close()
    print(f"\n✅ Benchmarking finished. File saved to {csv_filename}")

if __name__ == '__main__':
    main()