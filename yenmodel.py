from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import serial
import time
import numpy as np
import matplotlib.pyplot as plt


PORT = 'COM4'
BAUDRATE = 230400
READ_INTERVAL = 1  # saniye


def collect_initial_data(ser, duration=1):
    data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        if ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            try:
                val = float(line)
                data.append(val)
            except:
                pass
        else:
            time.sleep(0.01)
    return np.array(data).reshape(-1, 1)

def main():
    ser = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)

    initial_data = collect_initial_data(ser, duration=1)  # 1 sn veri topla

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(initial_data)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(scaled_train)


    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 5))

    try:
        while True:
            data = collect_initial_data(ser, READ_INTERVAL)
            if len(data) == 0:
                print("Veri gelmedi, tekrar dene...")
                continue

            data_np = np.array(data).reshape(-1, 1)
            scaled_data = scaler.transform(data_np)
            preds = model.predict(scaled_data)
            anomalies = preds == -1

            x = np.arange(len(data))

            ax.clear()
            ax.plot(x, data, 'b-', label='Veri')
            if anomalies.any():
                ax.scatter(x[anomalies], np.array(data)[anomalies], color='red', label='Anomali')
            ax.set_title(f'Anomali Tespiti ({READ_INTERVAL}s)')
            ax.set_xlabel('Data Noktası')
            ax.set_ylabel('Değer')
            ax.legend()
            plt.pause(0.05)

            print(f"Son {READ_INTERVAL} saniye veri sayısı: {len(data)}, Anomali sayısı: {np.sum(anomalies)}")

    except KeyboardInterrupt:
        print("Program durduruldu.")
    finally:
        ser.close()


if __name__ == "__main__":
    main()

