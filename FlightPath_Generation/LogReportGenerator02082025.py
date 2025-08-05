import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


def parse_log(file_path):
    """
    parse log file and return position records, telemetry and quality dataframes
    """
    records, telemetry, quality = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7 or parts[2] != '41':
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
                    quality.append({'quality_pct': float(parts[5])})
                except ValueError:
                    continue
    return pd.DataFrame(records), pd.DataFrame(telemetry), pd.DataFrame(quality)


def parse_flightpath(path_file):
    """
    parse flight path file into numpy array of shape (n,3)
    supports CSV files with columns x,y,z or x,z (assumes missing axis=0), or falls back to .dpt parsing
    """
    ext = os.path.splitext(path_file)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path_file)
        # three-column CSV
        if all(c in df.columns for c in ['x', 'y', 'z']):
            return df[['x', 'y', 'z']].values
        # two-column CSV: x,z → assume y=0
        elif all(c in df.columns for c in ['x', 'z']):
            x = df['x'].values
            z = df['z'].values
            y = np.zeros_like(x)
            return np.stack([x, y, z], axis=1)
        # two-column CSV: x,y → assume z=0
        elif all(c in df.columns for c in ['x', 'y']):
            x = df['x'].values
            y = df['y'].values
            z = np.zeros_like(x)
            return np.stack([x, y, z], axis=1)
        # fallback: first three numeric columns
        nums = df.select_dtypes(include=[np.number]).values
        if nums.shape[1] >= 2:
            if nums.shape[1] == 2:
                x, y = nums[:, 0], nums[:, 1]
                z = np.zeros_like(x)
                return np.stack([x, y, z], axis=1)
            return nums[:, :3]
        else:
            raise ValueError("CSV must contain at least two numeric columns for coordinates")
    # original .dpt-style parsing
    data = []
    with open(path_file, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 1 and ',' in vals[0]:
                vals = vals[0].split(',')
            if len(vals) >= 2:
                coords = (
                    list(map(float, vals[:3]))
                    if len(vals) >= 3
                    else list(map(float, vals[:2])) + [0.0]
                )
                data.append(coords)
    return np.array(data)


def point_to_segment_distance(P, A, B):
    ab = B - A
    denom = np.dot(ab, ab)
    if denom == 0:
        return np.linalg.norm(P - A)
    t = np.dot(P - A, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    c = A + t * ab
    return np.linalg.norm(P - c)


def deviations_along_path(positions, path):
    deviations = []
    for p in positions:
        dists = [point_to_segment_distance(p, path[i], path[i+1]) for i in range(len(path)-1)]
        deviations.append(min(dists))
    return np.array(deviations)


def infer_workspace_dims(coords, buffer_factor=0.05):
    """
    infer x, y, z limits with buffer around min/max
    """
    limits = {}
    for axis in ['x', 'y', 'z']:
        arr = coords[axis].dropna()
        if arr.empty:
            limits[axis] = (0, 0)
        else:
            mn, mx = arr.min(), arr.max()
            buff = (mx - mn) * buffer_factor
            limits[axis] = (mn - buff, mx + buff)
    return limits


def compute_dims(workspace_dims, inferred_dims):
    """
    build plotting dims: if user supplies a scalar, use (0, scalar);
    if a tuple/list of length 2, use it; if None, use inferred
    """
    dims = {}
    for axis in ['x', 'y', 'z']:
        wd = workspace_dims.get(axis) if workspace_dims else None
        if wd is None:
            dims[axis] = inferred_dims[axis]
        else:
            if isinstance(wd, (list, tuple)) and len(wd) == 2:
                dims[axis] = tuple(wd)
            elif isinstance(wd, (int, float)):
                dims[axis] = (0, wd)
            else:
                raise ValueError(f"workspace_dims[{axis}] must be None, a number, or a (min, max) tuple")
    return dims


def visualize_path_2d(coords, view='xy', dims=None, path=None):
    """
    create 2d path plot with time-based colour gradient, start/end markers,
    and optional intended path overlay in green (with black-dot waypoints labeled by index)
    """
    axis_map = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
    ax0, ax1 = axis_map[view]
    x = coords[ax0].values
    y = coords[ax1].values
    times = coords['datetime']
    t0 = times.min()
    t_secs = (times - t0).dt.total_seconds().values

    norm = plt.Normalize(t_secs.min(), t_secs.max())
    cmap = plt.cm.plasma

    # actual path line with time gradient
    points = np.vstack([x, y]).T
    segments = np.stack([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t_secs[:-1])
    lc.set_linewidth(2)
    ax.add_collection(lc)

    ax.scatter(x[0], y[0], marker='o', color='green', s=100, label='actual start')
    ax.scatter(x[-1], y[-1], marker='X', color='red', s=100, label='actual end')

    if path is not None:
        index_map = {'x': 0, 'y': 1, 'z': 2}
        px = path[:, index_map[ax0]]
        py = path[:, index_map[ax1]]
        # plot green line between waypoints
        ax.plot(px, py, '-', color='lime', linewidth=2, label='intended path')
        # plot black-dot waypoints
        ax.scatter(px, py, color='black', s=30, zorder=5)

    ax.set_xlabel(f'{ax0} (m)')
    ax.set_ylabel(f'{ax1} (m)')
    ax.set_title(f'{view}-view path (2D)')
    if dims:
        ax.set_xlim(dims[ax0])
        ax.set_ylim(dims[ax1])
    ax.legend()
    fig.colorbar(lc, ax=ax, label='time since start (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def visualize_path_3d(coords, dims=None, path=None):
    """
    create 3d path plot with time-based colour gradient, start/end markers,
    and optional intended path overlay in green (with black-dot waypoints labeled by index)
    """
    x = coords['x'].values
    y = coords['y'].values
    z = coords['z'].values
    times = coords['datetime']
    t0 = times.min()
    t_secs = (times - t0).dt.total_seconds().values

    norm = plt.Normalize(t_secs.min(), t_secs.max())
    cmap = plt.cm.plasma

    # actual trajectory line
    points = np.vstack([x, y, z]).T
    segments = np.stack([points[:-1], points[1:]], axis=1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(t_secs[:-1])
    lc.set_linewidth(2)
    ax.add_collection(lc)

    # actual start and end markers
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='actual start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='actual end')

    if path is not None:
        # overlay intended path line
        px, py, pz = path[:, 0], path[:, 1], path[:, 2]
        ax.plot(px, py, pz, '-', color='lime', linewidth=2, label='intended path')
        # black-dot waypoints
        ax.scatter(px, py, pz, color='black', s=30, depthshade=False)
        # label waypoints by index
        for i, (xi, yi, zi) in enumerate(zip(px, py, pz)):
            ax.text(xi, yi, zi, str(i), fontsize=8)

    if dims:
        ax.set_xlim(dims['x'])
        ax.set_ylim(dims['y'])
        ax.set_zlim(dims['z'])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title('3D path with time gradient')
    fig.colorbar(lc, ax=ax, pad=0.1, label='time since start (s)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def analyze_and_report_with_plots(file_path, output_path, path_file=None, workspace_dims=None):
    df, df_tel, df_qual = parse_log(file_path)
    coords = df.dropna(subset=['x', 'y', 'z']).reset_index(drop=True)

    inferred = infer_workspace_dims(coords)
    dims = compute_dims(workspace_dims or {}, inferred)

    intended_path = parse_flightpath(path_file) if path_file else None

    # compute metrics
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

    # 1d plots
    plt.figure(figsize=(10, 6))
    plt.plot(coords['datetime'], coords['x'], label='x', linewidth=2)
    plt.plot(coords['datetime'], coords['y'], label='y', linewidth=2)
    plt.plot(coords['datetime'], coords['z'], label='z', linewidth=2)
    plt.xlabel('time')
    plt.ylabel('position (m)')
    plt.title('position vs time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(shifts['shift_ms'], bins=30, edgecolor='black')
    plt.xlabel('latency (ms)')
    plt.ylabel('frequency')
    plt.title('latency distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.hist(update_rates, bins=30, edgecolor='black')
    plt.xlabel('update rate (hz)')
    plt.ylabel('frequency')
    plt.title('update rate distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # telemetry and quality
    if not df_tel.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(df_tel['rssi_dbm'].reset_index(drop=True), linewidth=1.5)
        plt.xlabel('sample index')
        plt.ylabel('rssi (dbm)')
        plt.title('rssi over samples')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(df_tel['voltage_v'].reset_index(drop=True), linewidth=1.5)
        plt.xlabel('sample index')
        plt.ylabel('voltage (v)')
        plt.title('voltage over samples')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if not df_qual.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(df_qual['quality_pct'].reset_index(drop=True), linewidth=1.5)
        plt.xlabel('sample index')
        plt.ylabel('quality (%)')
        plt.title('location quality over samples')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # path visualizations
    visualize_path_3d(coords, dims=dims, path=intended_path)
    for view in ['xy', 'xz', 'yz']:
        visualize_path_2d(coords, view=view, dims=dims, path=intended_path)

    # path accuracy metrics
    if intended_path is not None:
        devs = deviations_along_path(coords[['x', 'y', 'z']].values, intended_path)
        mean_dev = np.mean(devs)
        max_dev = np.max(devs)
        rms_dev = np.sqrt(np.mean(devs**2))
        pct95 = np.percentile(devs, 95)

        plt.figure(figsize=(10, 4))
        plt.plot(coords['datetime'], devs, linewidth=1.5)
        plt.xlabel('time')
        plt.ylabel('deviation (m)')
        plt.title('path deviation vs time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 4))
        plt.hist(devs, bins=30, edgecolor='black')
        plt.xlabel('deviation (m)')
        plt.ylabel('frequency')
        plt.title('path deviation distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # write markdown report
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

        rpt.write("## 5. Signal Quality\n")
        if volt_avg is not None:
            rpt.write(f"- voltage: avg {volt_avg:.2f} V, σ {volt_sd:.2f} V\n")
            rpt.write(f"- rssi: avg {rssi_avg:.1f} dBm, σ {rssi_sd:.1f} dBm\n")
            rpt.write(f"- location quality: avg {qual_avg:.1f} %, σ {qual_sd:.1f} %\n\n")
        else:
            rpt.write("- no telemetry/quality data found\n\n")

        rpt.write("## 6. Coordinate Resolution\n")
        rpt.write(f"- x: {res_x} m\n")
        rpt.write(f"- y: {res_y} m\n")
        rpt.write(f"- z: {res_z} m\n")

        if intended_path is not None:
            rpt.write("\n## Path Accuracy Metrics\n")
            rpt.write(f"- mean deviation: {mean_dev:.3f} m\n")
            rpt.write(f"- max deviation: {max_dev:.3f} m\n")
            rpt.write(f"- RMS deviation: {rms_dev:.3f} m\n")
            rpt.write(f"- 95th percentile deviation: {pct95:.3f} m\n")
    print(f"report and plots generated. report saved to {output_path}")


if __name__ == '__main__':
    workspace_dims = {'x': 6.0, 'y': 6.0, 'z': 5.0}
    log_file = '/Users/benjamindrury/Downloads/2025_08_01__152835__Marvelmind_log.csv'
    report_file = 'marvelmind_report.md'
    path_file = '/Users/benjamindrury/Downloads/waypoints.csv'
    analyze_and_report_with_plots(
        log_file,
        report_file,
        path_file=path_file,
        workspace_dims=workspace_dims
    )
