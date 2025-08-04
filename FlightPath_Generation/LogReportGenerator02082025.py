import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def parse_log(file_path):
    records, telemetry, quality = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            if parts[2] != '41':
                continue
            code = parts[3]
            ts = parts[0]
            if code in ['17', '129'] and len(parts) >= 11:
                try:
                    dt = datetime.strptime(ts, 'T%Y_%m_%d__%H%M%S_%f')
                except ValueError:
                    continue
                x, y, z, shift = parts[5], parts[6], parts[7], parts[10]
                records.append({
                    'datetime': dt,
                    'x': None if x.lower() == 'na' else float(x),
                    'y': None if y.lower() == 'na' else float(y),
                    'z': None if z.lower() == 'na' else float(z),
                    'shift_ms': None if shift.lower() == 'na' else float(shift)
                })
            elif code == '6' and len(parts) >= 7:
                try:
                    telemetry.append({
                        'voltage_v': float(parts[5]),
                        'rssi_dbm': float(parts[6])
                    })
                except ValueError:
                    continue
            elif code == '7' and len(parts) >= 6:
                try:
                    quality.append({
                        'quality_pct': float(parts[5])
                    })
                except ValueError:
                    continue
    return pd.DataFrame(records), pd.DataFrame(telemetry), pd.DataFrame(quality)


def parse_flightpath(path_file):
    data = []
    with open(path_file, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 1 and ',' in vals[0]:
                vals = vals[0].split(',')
            if len(vals) >= 2:
                coords = list(map(float, vals[:3])) if len(vals) >= 3 else list(map(float, vals[:2])) + [0.0]
                data.append(coords)
    return np.array(data)


def point_to_segment_distance(P, A, B):
    AB = B - A
    denom = np.dot(AB, AB)
    if denom == 0:
        return np.linalg.norm(P - A)
    t = np.dot(P - A, AB) / denom
    t = np.clip(t, 0.0, 1.0)
    C = A + t * AB
    return np.linalg.norm(P - C)


def deviations_along_path(positions, path):
    deviations = []
    for P in positions:
        dists = [point_to_segment_distance(P, path[i], path[i+1]) for i in range(len(path)-1)]
        deviations.append(min(dists))
    return np.array(deviations)


def analyze_and_report_with_plots(file_path, output_path, path_file=None):
    df, df_tel, df_qual = parse_log(file_path)
    coords = df.dropna(subset=['x', 'y', 'z'])
    shifts = df.dropna(subset=['shift_ms'])
    diffs = coords['datetime'].diff().dt.total_seconds().dropna()
    update_rates = 1 / diffs if not diffs.empty else pd.Series(dtype=float)

    stability = coords[['x', 'y', 'z']].std()
    total = len(df)
    avail = len(coords)
    availability_pct = avail / total * 100 if total else 0
    latency_avg = shifts['shift_ms'].mean()
    latency_sd = shifts['shift_ms'].std()
    update_avg = update_rates.mean() if not update_rates.empty else None
    update_sd = update_rates.std() if not update_rates.empty else None
    volt_avg = df_tel['voltage_v'].mean() if not df_tel.empty else None
    volt_sd = df_tel['voltage_v'].std() if not df_tel.empty else None
    rssi_avg = df_tel['rssi_dbm'].mean() if not df_tel.empty else None
    rssi_sd = df_tel['rssi_dbm'].std() if not df_tel.empty else None
    qual_avg = df_qual['quality_pct'].mean() if not df_qual.empty else None
    qual_sd = df_qual['quality_pct'].std() if not df_qual.empty else None
    def resolution(arr):
        vals = sorted(set(arr.dropna()))
        gaps = [j - i for i, j in zip(vals, vals[1:])]
        return min(gaps) if gaps else None
    res_x = resolution(coords['x'])
    res_y = resolution(coords['y'])
    res_z = resolution(coords['z'])

    # Define colors
    color_x, color_y, color_z = 'tab:blue', 'tab:orange', 'tab:green'
    arrow_color = 'tab:red'
    path_color = 'tab:purple'

    # Position vs Time
    plt.figure(figsize=(10, 6))
    plt.plot(coords['datetime'], coords['x'], label='X', color=color_x, linewidth=2)
    plt.plot(coords['datetime'], coords['y'], label='Y', color=color_y, linewidth=2)
    plt.plot(coords['datetime'], coords['z'], label='Z', color=color_z, linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Position (m)')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # Latency Distribution
    plt.figure(figsize=(8, 4))
    plt.hist(shifts['shift_ms'], bins=30, color='tab:cyan', edgecolor='black')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Frequency')
    plt.title('Latency Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Update Rate Distribution
    plt.figure(figsize=(8, 4))
    plt.hist(update_rates, bins=30, color='tab:olive', edgecolor='black')
    plt.xlabel('Update Rate (Hz)')
    plt.ylabel('Frequency')
    plt.title('Update Rate Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Signal Quality Over Samples
    if not df_tel.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(df_tel['rssi_dbm'].reset_index(drop=True), linewidth=1.5, color='tab:brown')
        plt.xlabel('Sample Index')
        plt.ylabel('RSSI (dBm)')
        plt.title('RSSI over Samples')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8, 4))
        plt.plot(df_tel['voltage_v'].reset_index(drop=True), linewidth=1.5, color='tab:gray')
        plt.xlabel('Sample Index')
        plt.ylabel('Voltage (V)')
        plt.title('Voltage over Samples')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    if not df_qual.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(df_qual['quality_pct'].reset_index(drop=True), linewidth=1.5, color='tab:pink')
        plt.xlabel('Sample Index')
        plt.ylabel('Quality (%)')
        plt.title('Location Quality over Samples')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # 3D Trajectory with Arrows
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords['x'], coords['y'], coords['z'], label='3D Path', color=path_color, linewidth=2, alpha=0.8)
    step3 = max(len(coords)//15, 1)
    for i in range(0, len(coords)-1, step3):
        ax.quiver(coords['x'].iloc[i], coords['y'].iloc[i], coords['z'].iloc[i],
                  coords['x'].iloc[i+1] - coords['x'].iloc[i],
                  coords['y'].iloc[i+1] - coords['y'].iloc[i],
                  coords['z'].iloc[i+1] - coords['z'].iloc[i],
                  length=0.1, normalize=True, color=arrow_color, alpha=0.8)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory with Direction Arrows')
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # 2D Views from Multiple Angles
    fig2, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].plot(coords['x'], coords['y'], '-o', markersize=4, linewidth=1.5, color=path_color)
    axs[0].set_xlabel('X (m)')
    axs[0].set_ylabel('Y (m)')
    axs[0].set_title('Top-down (XY)')
    axs[0].grid(True)
    axs[1].plot(coords['x'], coords['z'], '-o', markersize=4, linewidth=1.5, color=path_color)
    axs[1].set_xlabel('X (m)')
    axs[1].set_ylabel('Z (m)')
    axs[1].set_title('Side (XZ)')
    axs[1].grid(True)
    axs[2].plot(coords['y'], coords['z'], '-o', markersize=4, linewidth=1.5, color=path_color)
    axs[2].set_xlabel('Y (m)')
    axs[2].set_ylabel('Z (m)')
    axs[2].set_title('Front (YZ)')
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()

    # Path accuracy if requested
    if path_file:
        path = parse_flightpath(path_file)
        pos_arr = coords[['x', 'y', 'z']].values
        devs = deviations_along_path(pos_arr, path)
        mean_dev = np.mean(devs)
        max_dev = np.max(devs)
        rms_dev = np.sqrt(np.mean(devs**2))
        pct95 = np.percentile(devs, 95)
        plt.figure(figsize=(10, 4))
        plt.plot(coords['datetime'].iloc[1:], devs, linewidth=1.5, color='tab:red')
        plt.xlabel('Time')
        plt.ylabel('Deviation (m)')
        plt.title('Path Deviation vs Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(8, 4))
        plt.hist(devs, bins=30, color='tab:purple', edgecolor='black')
        plt.xlabel('Deviation (m)')
        plt.ylabel('Frequency')
        plt.title('Path Deviation Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    with open(output_path, 'w') as rpt:
        rpt.write("# Marvelmind System Performance Report\n\n")
        rpt.write("## 1. Position Stability (σ)\n")
        rpt.write(f"- x: {stability['x']:.3f} m\n")
        rpt.write(f"- y: {stability['y']:.3f} m\n")
        rpt.write(f"- z: {stability['z']:.3f} m\n\n")
        rpt.write("## 2. Data Availability\n")
        rpt.write(f"- {availability_pct:.2f}% ({avail}/{total})\n\n")
        rpt.write("## 3. Latency\n")
        rpt.write(f"- avg: {latency_avg:.1f} ms\n")
        rpt.write(f"- σ: {latency_sd:.1f} ms\n\n")
        rpt.write("## 4. Update Rate\n")
        rpt.write(f"- avg: {update_avg:.1f} Hz\n")
        rpt.write(f"- σ: {update_sd:.1f} Hz\n\n")
        rpt.write("## 8. Signal Quality\n")
        if volt_avg is not None:
            rpt.write(f"- voltage: avg {volt_avg:.2f} V, σ {volt_sd:.2f} V\n")
            rpt.write(f"- rssi: avg {rssi_avg:.1f} dBm, σ {rssi_sd:.1f} dBm\n")
            rpt.write(f"- location quality: avg {qual_avg:.1f} %, σ {qual_sd:.1f} %\n\n")
        else:
            rpt.write("- no telemetry/quality data found\n\n")
        rpt.write("## 10. Coordinate Resolution\n")
        rpt.write(f"- x: {res_x} m\n")
        rpt.write(f"- y: {res_y} m\n")
        rpt.write(f"- z: {res_z} m\n")
        if path_file:
            rpt.write("\n## Path Accuracy Metrics\n")
            rpt.write(f"- mean deviation: {mean_dev:.3f} m\n")
            rpt.write(f"- max deviation: {max_dev:.3f} m\n")
            rpt.write(f"- RMS deviation: {rms_dev:.3f} m\n")
            rpt.write(f"- 95th percentile deviation: {pct95:.3f} m\n")
    print(f"Report and plots generated. Report saved to {output_path}")

if __name__ == "__main__":
    log_file = "/Users/benjamindrury/Downloads/2025_08_01__152835__Marvelmind_log.csv"
    report_file = "/mnt/data/marvelmind_report.md"
    analyze_and_report_with_plots(log_file, report_file, path_file=None)
