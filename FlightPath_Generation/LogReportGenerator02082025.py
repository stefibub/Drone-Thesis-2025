import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.ticker as ticker


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
        if all(c in df.columns for c in ['x', 'y', 'z']):
            return df[['x', 'y', 'z']].values
        elif all(c in df.columns for c in ['x', 'z']):
            x, z = df['x'].values, df['z'].values
            y = np.zeros_like(x)
            return np.stack([x, y, z], axis=1)
        elif all(c in df.columns for c in ['x', 'y']):
            x, y = df['x'].values, df['y'].values
            z = np.zeros_like(x)
            return np.stack([x, y, z], axis=1)
        nums = df.select_dtypes(include=[np.number]).values
        if nums.shape[1] >= 2:
            if nums.shape[1] == 2:
                x, y = nums[:, 0], nums[:, 1]
                z = np.zeros_like(x)
                return np.stack([x, y, z], axis=1)
            return nums[:, :3]
        raise ValueError("CSV must contain at least two numeric columns for coordinates")
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
        dists = [point_to_segment_distance(p, path[i], path[i + 1]) for i in range(len(path) - 1)]
        deviations.append(min(dists))
    return np.array(deviations)


def trim_to_waypoints(
    coords,
    intended_path,
    r_wp=0.5,
    z_thresh=0.9
):
    """
    Crop coords so that we keep only:
     • after takeoff (z > z_thresh),
     • from first entry within r_wp of the first waypoint,
     • up until just before landing begins (z drops below z_thresh) or just after final waypoint.

    Parameters
    ----------
    coords : pd.DataFrame with columns ['x','y','z', …]
    intended_path : np.ndarray of shape (N,3), your planned waypoints
    r_wp : float
      radius around waypoints (m) to detect “arrival”
    z_thresh : float
      altitude (m) above which takeoff is considered complete / landing not started

    Returns
    -------
    pd.DataFrame
    """
    P = coords[['x','y','z']].values
    wp0, wpn = intended_path[0], intended_path[-1]

    # 1) drop before takeoff altitude
    above = coords['z'].values > z_thresh
    if above.any():
        idx_takeoff = np.argmax(above)
    else:
        idx_takeoff = 0

    # 2) circle‐around‐first‐WP trim
    d0 = np.linalg.norm(P - wp0, axis=1)
    # only look after takeoff
    rel0 = d0[idx_takeoff:]
    rel_start = np.argmax(rel0 <= r_wp)
    start_idx = idx_takeoff + rel_start

    # 3) determine end‐of‐flight
    #   a) last time above z_thresh
    if above.any():
        idx_land = len(above) - np.argmax(above[::-1]) - 1
    else:
        idx_land = len(coords) - 1

    #   b) last time within r_wp of final WP
    dn = np.linalg.norm(P - wpn, axis=1)
    last_wp_hits = np.where(dn <= r_wp)[0]
    if last_wp_hits.size:
        idx_wp_end = last_wp_hits[-1]
    else:
        idx_wp_end = idx_land

    # choose the later cut point (so we include full path to last WP)
    end_idx = max(idx_wp_end + 1, idx_land + 1)

    return coords.iloc[start_idx:end_idx].reset_index(drop=True)





def infer_workspace_dims(coords, buffer_factor=0.05):
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


def visualize_path_2d(coords, view='xy', dims=None, path=None, use_elapsed=False):
    axis_map = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
    ax0, ax1 = axis_map[view]
    x, y = coords[ax0].values, coords[ax1].values

    if use_elapsed:
        t_secs = coords['elapsed_s'].values
    else:
        t0 = coords['datetime'].min()
        t_secs = (coords['datetime'] - t0).dt.total_seconds().values

    norm = plt.Normalize(t_secs.min(), t_secs.max())
    cmap = plt.cm.plasma

    points = np.vstack([x, y]).T
    segments = np.stack([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(t_secs[:-1])
    ax.add_collection(lc)

    ax.scatter(x[0], y[0], marker='o', color='green', s=100, label='start')
    ax.scatter(x[-1], y[-1], marker='X', color='red', s=100, label='end')

    if path is not None:
        idx = {'x': 0, 'y': 1, 'z': 2}
        px, py = path[:, idx[ax0]], path[:, idx[ax1]]
        ax.plot(px, py, '-', color='lime', linewidth=2, label='intended')
        ax.scatter(px, py, color='black', s=30, zorder=5)

    ax.set_xlabel(f'{ax0} (m)')
    ax.set_ylabel(f'{ax1} (m)')
    ax.set_title(f'{view}-view path (2D)')
    if dims:
        ax.set_xlim(dims[ax0]); ax.set_ylim(dims[ax1])
    ax.legend()
    fig.colorbar(lc, ax=ax, label='time since start (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def visualize_path_3d(coords, dims=None, path=None, use_elapsed=False):
    x, y, z = coords['x'].values, coords['y'].values, coords['z'].values

    if use_elapsed:
        t_secs = coords['elapsed_s'].values
    else:
        t0 = coords['datetime'].min()
        t_secs = (coords['datetime'] - t0).dt.total_seconds().values

    norm = plt.Normalize(t_secs.min(), t_secs.max())
    cmap = plt.cm.plasma

    points = np.vstack([x, y, z]).T
    segments = np.stack([points[:-1], points[1:]], axis=1)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    lc = Line3DCollection(segments, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(t_secs[:-1])
    ax.add_collection(lc)

    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='end')

    if path is not None:
        px, py, pz = path[:, 0], path[:, 1], path[:, 2]
        ax.plot(px, py, pz, '-', color='lime', linewidth=2, label='intended')
        ax.scatter(px, py, pz, color='black', s=30, depthshade=False)
        for i, (xi, yi, zi) in enumerate(zip(px, py, pz)):
            ax.text(xi, yi, zi, str(i), fontsize=8)

    if dims:
        ax.set_xlim(dims['x']); ax.set_ylim(dims['y']); ax.set_zlim(dims['z'])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('3D path with time gradient')
    fig.colorbar(lc, ax=ax, pad=0.1, label='time since start (s)')
    ax.legend()
    plt.tight_layout()
    plt.show()


def compute_path_accuracy(coords, intended_path):
    """
    Compute cross-track deviations e_i for each logged point, then return
    the array of deviations plus summary metrics.
    """
    # extract only x,y from your logged points
    P_xy = coords[['x','y']].values

    # likewise project your intended_path onto the XY-plane:
    #   if intended_path is an (N×3) array, we just take the first two cols
    intended_xy = intended_path[:, :2]

    # compute deviations in 2D
    e_xy = deviations_along_path(P_xy, intended_xy)

    # summary stats on the 2D deviations
    metrics_xy = {
        'mean'   : np.mean(e_xy),
        'rms'    : np.sqrt(np.mean(e_xy**2)),
        'max'    : np.max(e_xy),
        'pct95'  : np.percentile(e_xy, 95),
    }
    return e_xy, metrics_xy

def compute_waypoint_metrics(coords, intended_path, tol=0.10):
    """
    For each waypoint, find the closest logged point in the XY plane:
    record horizontal error, timestamp, and whether error <= tol.
    Returns list of dicts and overall success_rate (%).
    """
    # extract only the horizontal coordinates
    P_xy = coords[['x','y']].values

    wp_results = []
    hits = 0
    for i, wp in enumerate(intended_path):
        # only take the x,y of the waypoint
        wp_xy = wp[:2]
        # compute 2D distance
        d_xy = np.linalg.norm(P_xy - wp_xy, axis=1)

        idx = np.argmin(d_xy)
        err = float(d_xy[idx])
        t   = coords['datetime'].iloc[idx]
        hit = err <= tol
        hits += int(hit)

        wp_results.append({
            'index'  : i,
            'error_m': err,
            'time'   : t,
            'hit'    : hit
        })

    success_rate = hits / len(intended_path) * 100.0
    return wp_results, success_rate




def analyze_and_report_with_plots(file_path, output_path, path_file=None, workspace_dims=None):
    df, df_tel, df_qual = parse_log(file_path)
    # drop any rows with missing x,y,z
    coords = df.dropna(subset=['x', 'y', 'z']).reset_index(drop=True)

    # if we have an intended path, trim out takeoff & landing
    if path_file:
        intended_path = parse_flightpath(path_file)
        coords = trim_to_waypoints(coords, intended_path)
    else:
        intended_path = None

    # compute elapsed seconds since start of trimmed segment
    t0 = coords['datetime'].min()
    coords['elapsed_s'] = (coords['datetime'] - t0).dt.total_seconds()

    # prepare MM:SS formatter
    def format_mmss(x, pos):
        m = int(x) // 60
        s = int(x) % 60
        return f"{m:02d}:{s:02d}"
    mmss_formatter = ticker.FuncFormatter(format_mmss)

        # --- PATH-FOLLOWING ACCURACY ---
    if intended_path is not None:
        devs, path_metrics = compute_path_accuracy(coords, intended_path)
        # add final check
        within_bound = np.mean(devs <= 0.10) > 0.5

        # plot deviation-vs-time again with smoothed if desired
        # …

    # --- WAYPOINT METRICS ---
    if intended_path is not None:
        wp_results, wp_success = compute_waypoint_metrics(coords, intended_path, tol=0.10)
        # optional: make a small DataFrame for easy viewing
        wp_df = pd.DataFrame(wp_results)




    # infer plotting dims
    inferred = infer_workspace_dims(coords)
    dims     = compute_dims(workspace_dims or {}, inferred)

    # --- 1D position vs flight time ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coords['elapsed_s'], coords['x'], label='x', linewidth=2)
    ax.plot(coords['elapsed_s'], coords['y'], label='y', linewidth=2)
    ax.plot(coords['elapsed_s'], coords['z'], label='z', linewidth=2)
    ax.set_xlabel('time (MM:SS)')
    ax.set_ylabel('position (m)')
    ax.set_title('position vs flight time')
    ax.legend(); ax.grid(True)
    ax.xaxis.set_major_formatter(mmss_formatter)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    plt.tight_layout(); plt.show()

    # --- latency distribution ---
    plt.figure(figsize=(8, 4))
    plt.hist(df.dropna(subset=['shift_ms'])['shift_ms'], bins=30, edgecolor='black')
    plt.xlabel('latency (ms)'); plt.ylabel('frequency'); plt.title('latency distribution')
    plt.grid(True); plt.tight_layout(); plt.show()

    # --- update rate distribution ---
    diffs = coords['datetime'].diff().dt.total_seconds().dropna()
    update_rates = 1 / diffs if not diffs.empty else pd.Series(dtype=float)
    plt.figure(figsize=(8, 4))
    plt.hist(update_rates, bins=30, edgecolor='black')
    plt.xlabel('update rate (Hz)'); plt.ylabel('frequency'); plt.title('update rate distribution')
    plt.grid(True); plt.tight_layout(); plt.show()

    # --- telemetry & quality over samples ---
    if not df_tel.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(df_tel['rssi_dbm'].reset_index(drop=True), linewidth=1.5)
        plt.xlabel('sample index'); plt.ylabel('rssi (dBm)'); plt.title('RSSI over samples')
        plt.grid(True); plt.tight_layout(); plt.show()

        plt.figure(figsize=(8, 4))
        plt.plot(df_tel['voltage_v'].reset_index(drop=True), linewidth=1.5)
        plt.xlabel('sample index'); plt.ylabel('voltage (V)'); plt.title('voltage over samples')
        plt.grid(True); plt.tight_layout(); plt.show()

    if not df_qual.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(df_qual['quality_pct'].reset_index(drop=True), linewidth=1.5)
        plt.xlabel('sample index'); plt.ylabel('quality (%)'); plt.title('location quality over samples')
        plt.grid(True); plt.tight_layout(); plt.show()

    # --- 3D + 2D path visualizations ---
    visualize_path_3d(coords, dims=dims, path=intended_path, use_elapsed=True)
    for view in ['xy', 'xz', 'yz']:
        visualize_path_2d(coords, view=view, dims=dims, path=intended_path, use_elapsed=True)

    # --- path deviation metrics & plot ---
    if intended_path is not None:
        devs       = deviations_along_path(coords[['x', 'y', 'z']].values, intended_path)
        mean_dev   = np.mean(devs)
        max_dev    = np.max(devs)
        rms_dev    = np.sqrt(np.mean(devs**2))
        pct95      = np.percentile(devs, 95)
        window     = 5
        devs_smooth = pd.Series(devs).rolling(window, center=True, min_periods=1).mean().values

        fig, ax = plt.subplots(figsize=(10, 4))
        #ax.plot(coords['elapsed_s'], devs,       alpha=0.3, label='raw')
        ax.plot(coords['elapsed_s'], devs_smooth, linewidth=1.5, label='smoothed')
        ax.set_xlabel('time (MM:SS)'); ax.set_ylabel('deviation (m)')
        ax.set_title('path deviation vs flight time')
        ax.legend(); ax.grid(True)
        ax.xaxis.set_major_formatter(mmss_formatter)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
        plt.tight_layout(); plt.show()

        plt.figure(figsize=(8, 4))
        plt.hist(devs, bins=30, edgecolor='black')
        plt.xlabel('deviation (m)'); plt.ylabel('frequency'); plt.title('path deviation distribution')
        plt.grid(True); plt.tight_layout(); plt.show()

    # --- write markdown report ---


    with open(output_path, 'w') as rpt:
        rpt.write("# Marvelmind System Performance Report\n\n")
        if intended_path is not None:
            rpt.write("## Continuous Path-Following Accuracy\n")
            rpt.write(f"- Mean error:      {path_metrics['mean']:.3f} m\n")
            rpt.write(f"- RMS error:       {path_metrics['rms']:.3f} m\n")
            rpt.write(f"- Max error:       {path_metrics['max']:.3f} m\n")
            rpt.write(f"- 95th‐pct error:  {path_metrics['pct95']:.3f} m\n")
            rpt.write(f"- Within ±0.10 m:  {'✔' if within_bound else '✘'}\n\n")

            # just before writing the waypoint section:
            tol = 0.10
            # Count hits directly from your results list
            wp_hits   = sum(1 for r in wp_results if r['error_m'] <= tol)
            wp_total  = len(wp_results)
            wp_success = 100 * wp_hits / wp_total

            # Then write them out verbatim
            rpt.write("## Waypoint Accuracy & Success Rate\n")
            rpt.write(
                f"- Waypoints hit within {tol:.2f} m: {wp_success:.1f}% "
                f"({wp_hits}/{wp_total})\n\n"
)

            # Header with alignment
            rpt.write("| WP # | Error (m) |   Time   | Hit? |\n")
            rpt.write("|:----:|:---------:|:--------:|:----:|\n")

            # Rows with fixed-width, centered fields
            for r in wp_results:
                idx  = r['index']
                err  = r['error_m']
                tstr = r['time'].strftime("%H:%M:%S")
                hit  = '✔' if r['hit'] else '✘'
                rpt.write(f"| {idx:^4} | {err:^9.3f} | {tstr:^8} | {hit:^4} |\n")


        rpt.write("\n## Position Stability (σ)\n")
        rpt.write(f"- x: {coords['x'].std():.3f} m\n")
        rpt.write(f"- y: {coords['y'].std():.3f} m\n")
        rpt.write(f"- z: {coords['z'].std():.3f} m\n\n")

        total, avail = len(df), len(coords)
        rpt.write("## Data Availability\n")
        rpt.write(f"- {avail/total*100:.2f}% ({avail}/{total})\n\n")

        if not df.dropna(subset=['shift_ms']).empty:
            rpt.write("## Latency\n")
            rpt.write(f"- avg: {df['shift_ms'].mean():.1f} ms\n")
            rpt.write(f"- σ: {df['shift_ms'].std():.1f} ms\n\n")

        if not update_rates.empty:
            rpt.write("## Update Rate\n")
            rpt.write(f"- avg: {update_rates.mean():.1f} Hz\n")
            rpt.write(f"- σ: {update_rates.std():.1f} Hz\n\n")

        if not df_tel.empty:
            rpt.write("## Signal Quality\n")
            rpt.write(f"- voltage: avg {df_tel['voltage_v'].mean():.2f} V, σ {df_tel['voltage_v'].std():.2f} V\n")
            rpt.write(f"- rssi:    avg {df_tel['rssi_dbm'].mean():.1f} dBm, σ {df_tel['rssi_dbm'].std():.1f} dBm\n")
            if not df_qual.empty:
                rpt.write(f"- location quality: avg {df_qual['quality_pct'].mean():.1f} %, σ {df_qual['quality_pct'].std():.1f} %\n\n")

        def resolution(arr):
            vals = sorted(set(arr.dropna()))
            gaps = [j - i for i, j in zip(vals, vals[1:])]
            return min(gaps) if gaps else None

        rpt.write("## Coordinate Resolution\n")
        rpt.write(f"- x: {resolution(coords['x'])} m\n")
        rpt.write(f"- y: {resolution(coords['y'])} m\n")
        rpt.write(f"- z: {resolution(coords['z'])} m\n\n")

        if intended_path is not None:
            rpt.write("## Path Accuracy Metrics\n")
            rpt.write(f"- mean deviation: {mean_dev:.3f} m\n")
            rpt.write(f"- max deviation:  {max_dev:.3f} m\n")
            rpt.write(f"- RMS deviation:  {rms_dev:.3f} m\n")
            rpt.write(f"- 95th pct dev.:  {pct95:.3f} m\n")
    print(f"report and plots generated. report saved to {output_path}")


if __name__ == '__main__':
    workspace_dims = {'x': 6.0, 'y': 6.0, 'z': 2.0}
    log_file    = '/Users/benjamindrury/Desktop/Thesis Results/3dLawnmowerTest/3dLawnmower_Test/2025_08_06__194047__Marvelmind_logFlight_3DSpiral.csv'
    report_file = ('marvelmind_report_3dLawnmower_Test1.md')
    path_file   = '/Users/benjamindrury/PycharmProjects/Drone-Thesis-2025/FlightPaths/CsvFiles/mission_path_lawnmower_3d_20250805_232642.csv'

    analyze_and_report_with_plots(
        log_file,
        report_file,
        path_file=path_file,
        workspace_dims=workspace_dims
    )
