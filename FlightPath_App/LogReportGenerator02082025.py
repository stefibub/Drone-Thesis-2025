# LogReportGenerator.py
# Integration-ready report generator for Dash.
# Key API: generate_report_bundle(log_bytes: bytes, flightpath_bytes: Optional[bytes], workspace_dims: Optional[dict]) -> dict

from __future__ import annotations

import io
import os
import base64
import zipfile
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# Use headless backend for server environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.ticker as ticker
import base64, json, numpy as np, pandas as pd
from dataclasses import is_dataclass, asdict
import datetime as dt


# ------------------------------- Parsing -------------------------------

def _ensure_text(v):
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8")
        except Exception:
            return base64.b64encode(v).decode("ascii")
    return v if isinstance(v, str) else str(v)

def _ensure_b64(v):
    if isinstance(v, str):
        return v
    if isinstance(v, (bytes, bytearray)):
        return base64.b64encode(v).decode("ascii")
    return _ensure_text(v)

def make_json_safe(o):
    if isinstance(o, (bytes, bytearray)):
        return base64.b64encode(o).decode("ascii")
    if is_dataclass(o):
        return make_json_safe(asdict(o))
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.DataFrame):
        return o.to_dict(orient="records")
    if isinstance(o, pd.Series):
        return o.to_list()
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, dict):
        return {str(k): make_json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [make_json_safe(v) for v in o]
    return o


def parse_log_bytes(log_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse Marvelmind log content from bytes.
    Returns: (records_df, telemetry_df, quality_df)
    """
    records, telemetry, quality = [], [], []
    f = io.StringIO(log_bytes.decode("utf-8", errors="ignore"))
    for raw in f:
        parts = raw.strip().split(',')
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
            try_float = lambda v: None if str(v).strip().lower() == 'na' else float(v)
            records.append({
                'datetime': dt,
                'x': try_float(x),
                'y': try_float(y),
                'z': try_float(z),
                'shift_ms': try_float(shift),
            })

        elif code == '6' and len(parts) >= 7:
            try:
                telemetry.append({
                    'voltage_v': float(parts[5]),
                    'rssi_dbm': float(parts[6]),
                })
            except ValueError:
                continue

        elif code == '7' and len(parts) >= 6:
            try:
                quality.append({'quality_pct': float(parts[5])})
            except ValueError:
                continue

    return pd.DataFrame(records), pd.DataFrame(telemetry), pd.DataFrame(quality)


def parse_flightpath_bytes(fp_bytes: bytes) -> np.ndarray:
    """
    Parse a flightpath file (CSV or DPT-like text) provided as bytes.
    Returns an (N,3) numpy array of XYZ.
    """
    # Try CSV first with pandas
    try:
        df = pd.read_csv(io.BytesIO(fp_bytes))
        # Preferred: explicit x,y,z columns
        if all(c in df.columns for c in ['x', 'y', 'z']):
            return df[['x', 'y', 'z']].to_numpy(dtype=float)
        # Fallbacks to any 2+ numeric columns
        nums = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if nums.shape[1] >= 3:
            return nums[:, :3]
        if nums.shape[1] == 2:
            x, y = nums[:, 0], nums[:, 1]
            z = np.zeros_like(x)
            return np.stack([x, y, z], axis=1)
        raise ValueError("CSV must contain at least two numeric columns for coordinates")
    except Exception:
        pass

    # Fallback: simple DPT-like free text (space or comma separated)
    data: List[List[float]] = []
    f = io.StringIO(fp_bytes.decode("utf-8", errors="ignore"))
    for line in f:
        vals = line.strip().split()
        if len(vals) == 1 and ',' in vals[0]:
            vals = vals[0].split(',')
        if len(vals) >= 2:
            try:
                coords = (
                    list(map(float, vals[:3]))
                    if len(vals) >= 3
                    else list(map(float, vals[:2])) + [0.0]
                )
                data.append(coords)
            except ValueError:
                continue
    arr = np.array(data, dtype=float)
    if arr.ndim == 2 and arr.shape[1] >= 2:
        if arr.shape[1] == 2:
            z = np.zeros(arr.shape[0], dtype=float)
            arr = np.column_stack([arr[:, 0], arr[:, 1], z])
        else:
            arr = arr[:, :3]
        return arr
    raise ValueError("Unable to parse flightpath bytes into coordinates")


# ------------------------------- Geometry / Metrics -------------------------------

def point_to_segment_distance(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    ab = B - A
    denom = float(np.dot(ab, ab))
    if denom == 0.0:
        return float(np.linalg.norm(P - A))
    t = float(np.dot(P - A, ab) / denom)
    t = np.clip(t, 0.0, 1.0)
    C = A + t * ab
    return float(np.linalg.norm(P - C))


def deviations_along_path(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """
    points: (M,2) or (M,3)
    polyline: (N,2) or (N,3)
    Returns minimal distance from each point to polyline segments.
    """
    devs = []
    for p in points:
        seg_dists = [point_to_segment_distance(p, polyline[i], polyline[i + 1])
                     for i in range(len(polyline) - 1)]
        devs.append(min(seg_dists) if seg_dists else 0.0)
    return np.asarray(devs, dtype=float)


def trim_to_waypoints(coords: pd.DataFrame, intended_path: np.ndarray,
                      r_wp: float = 0.5, z_thresh: float = 0.9) -> pd.DataFrame:
    """
    Keep data after takeoff (z>z_thresh), from first time within r_wp of first WP,
    until last touch of final WP or when z drops below threshold.
    """
    P = coords[['x', 'y', 'z']].to_numpy(dtype=float)
    wp0, wpn = intended_path[0], intended_path[-1]

    above = coords['z'].to_numpy(dtype=float) > z_thresh
    idx_takeoff = int(np.argmax(above)) if above.any() else 0

    d0 = np.linalg.norm(P - wp0, axis=1)
    rel0 = d0[idx_takeoff:]
    rel_start = int(np.argmax(rel0 <= r_wp)) if (rel0 <= r_wp).any() else 0
    start_idx = idx_takeoff + rel_start

    if above.any():
        idx_land = len(above) - int(np.argmax(above[::-1])) - 1
    else:
        idx_land = len(coords) - 1

    dn = np.linalg.norm(P - wpn, axis=1)
    last_wp_hits = np.where(dn <= r_wp)[0]
    idx_wp_end = int(last_wp_hits[-1]) if last_wp_hits.size else idx_land

    end_idx = max(idx_wp_end + 1, idx_land + 1)
    return coords.iloc[start_idx:end_idx].reset_index(drop=True)


def infer_workspace_dims(coords: pd.DataFrame, buffer_factor: float = 0.05) -> Dict[str, Tuple[float, float]]:
    limits: Dict[str, Tuple[float, float]] = {}
    for axis in ['x', 'y', 'z']:
        arr = coords[axis].dropna()
        if arr.empty:
            limits[axis] = (0.0, 0.0)
        else:
            mn, mx = float(arr.min()), float(arr.max())
            buff = (mx - mn) * buffer_factor
            limits[axis] = (mn - buff, mx + buff)
    return limits


def compute_dims(workspace_dims: Optional[Dict], inferred_dims: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
    """
    workspace_dims: dict like {'x': (min,max) or length, 'y': ..., 'z': ...} or None
    """
    dims: Dict[str, Tuple[float, float]] = {}
    for axis in ['x', 'y', 'z']:
        wd = (workspace_dims or {}).get(axis)
        if wd is None:
            dims[axis] = inferred_dims[axis]
        else:
            if isinstance(wd, (list, tuple)) and len(wd) == 2:
                dims[axis] = (float(wd[0]), float(wd[1]))
            elif isinstance(wd, (int, float)):
                dims[axis] = (0.0, float(wd))
            else:
                raise ValueError(f"workspace_dims[{axis}] must be None, a number, or a (min,max) tuple")
    return dims


# ------------------------------- Plot helpers -------------------------------

def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')


def _mmss_formatter():
    def format_mmss(x, _pos):
        m = int(x) // 60
        s = int(x) % 60
        return f"{m:02d}:{s:02d}"
    return ticker.FuncFormatter(format_mmss)


def fig_position_vs_time(coords: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(coords['elapsed_s'], coords['x'], label='x', linewidth=2)
    ax.plot(coords['elapsed_s'], coords['y'], label='y', linewidth=2)
    ax.plot(coords['elapsed_s'], coords['z'], label='z', linewidth=2)
    ax.set_xlabel('time (MM:SS)')
    ax.set_ylabel('position (m)')
    ax.set_title('position vs flight time')
    ax.legend(); ax.grid(True)
    ax.xaxis.set_major_formatter(_mmss_formatter())
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    return fig


def fig_latency_hist(df: pd.DataFrame) -> Optional[plt.Figure]:
    vals = df.dropna(subset=['shift_ms'])['shift_ms']
    if vals.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals, bins=30, edgecolor='black')
    ax.set_xlabel('latency (ms)'); ax.set_ylabel('frequency'); ax.set_title('latency distribution')
    ax.grid(True)
    return fig


def fig_update_rate_hist(coords: pd.DataFrame) -> Optional[plt.Figure]:
    diffs = coords['datetime'].diff().dt.total_seconds().dropna()
    if diffs.empty:
        return None
    update_rates = 1.0 / diffs
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(update_rates, bins=30, edgecolor='black')
    ax.set_xlabel('update rate (Hz)'); ax.set_ylabel('frequency'); ax.set_title('update rate distribution')
    ax.grid(True)
    return fig


def fig_signal(df_tel: pd.DataFrame) -> Optional[plt.Figure]:
    if df_tel.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_tel['rssi_dbm'].reset_index(drop=True), linewidth=1.5)
    ax.set_xlabel('sample index'); ax.set_ylabel('rssi (dBm)'); ax.set_title('RSSI over samples')
    ax.grid(True)
    return fig


def fig_voltage(df_tel: pd.DataFrame) -> Optional[plt.Figure]:
    if df_tel.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_tel['voltage_v'].reset_index(drop=True), linewidth=1.5)
    ax.set_xlabel('sample index'); ax.set_ylabel('voltage (V)'); ax.set_title('voltage over samples')
    ax.grid(True)
    return fig


def fig_quality(df_qual: pd.DataFrame) -> Optional[plt.Figure]:
    if df_qual.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_qual['quality_pct'].reset_index(drop=True), linewidth=1.5)
    ax.set_xlabel('sample index'); ax.set_ylabel('quality (%)'); ax.set_title('location quality over samples')
    ax.grid(True)
    return fig


def fig_path_3d(coords: pd.DataFrame, dims: Dict[str, Tuple[float, float]],
                intended: Optional[np.ndarray], use_elapsed: bool = True) -> plt.Figure:
    x, y, z = coords['x'].to_numpy(float), coords['y'].to_numpy(float), coords['z'].to_numpy(float)
    t_secs = coords['elapsed_s'].to_numpy(float) if use_elapsed else (coords['datetime'] - coords['datetime'].min()).dt.total_seconds().to_numpy(float)

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

    if intended is not None and intended.size:
        px, py, pz = intended[:, 0], intended[:, 1], intended[:, 2]
        ax.plot(px, py, pz, '-', color='lime', linewidth=2, label='intended')
        ax.scatter(px, py, pz, color='black', s=20, depthshade=False)

    if dims:
        ax.set_xlim(dims['x']); ax.set_ylim(dims['y']); ax.set_zlim(dims['z'])
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title('3D path with time gradient')
    fig.colorbar(lc, ax=ax, pad=0.1, label='time since start (s)')
    ax.legend()
    return fig


def fig_path_2d(coords: pd.DataFrame, view: str,
                dims: Dict[str, Tuple[float, float]], intended: Optional[np.ndarray],
                use_elapsed: bool = True) -> plt.Figure:
    axis_map = {'xy': ('x', 'y'), 'xz': ('x', 'z'), 'yz': ('y', 'z')}
    ax0, ax1 = axis_map[view]
    a = coords[ax0].to_numpy(float)
    b = coords[ax1].to_numpy(float)
    t_secs = coords['elapsed_s'].to_numpy(float) if use_elapsed else (coords['datetime'] - coords['datetime'].min()).dt.total_seconds().to_numpy(float)

    norm = plt.Normalize(t_secs.min(), t_secs.max())
    cmap = plt.cm.plasma

    pts = np.vstack([a, b]).T
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    fig, ax = plt.subplots(figsize=(8, 6))
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2)
    lc.set_array(t_secs[:-1])
    ax.add_collection(lc)

    ax.scatter(a[0], b[0], marker='o', color='green', s=80, label='start')
    ax.scatter(a[-1], b[-1], marker='X', color='red', s=80, label='end')

    if intended is not None and intended.size:
        idx = {'x': 0, 'y': 1, 'z': 2}
        px, py = intended[:, idx[ax0]], intended[:, idx[ax1]]
        ax.plot(px, py, '-', color='lime', linewidth=2, label='intended')
        ax.scatter(px, py, color='black', s=20, zorder=5)

    ax.set_xlabel(f'{ax0} (m)'); ax.set_ylabel(f'{ax1} (m)')
    ax.set_title(f'{view}-view path (2D)')
    if dims:
        ax.set_xlim(dims[ax0]); ax.set_ylim(dims[ax1])
    ax.legend()
    fig.colorbar(lc, ax=ax, label='time since start (s)')
    ax.grid(True)
    return fig


def fig_deviation_time(coords: pd.DataFrame, intended: np.ndarray) -> plt.Figure:
    P = coords[['x', 'y', 'z']].to_numpy(float)
    devs = deviations_along_path(P, intended)
    window = 5
    devs_smooth = pd.Series(devs).rolling(window, center=True, min_periods=1).mean().to_numpy(float)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(coords['elapsed_s'], devs_smooth, linewidth=1.5, label='smoothed deviation')
    ax.set_xlabel('time (MM:SS)'); ax.set_ylabel('deviation (m)')
    ax.set_title('path deviation vs flight time')
    ax.legend(); ax.grid(True)
    ax.xaxis.set_major_formatter(_mmss_formatter())
    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
    return fig


def fig_deviation_hist(coords: pd.DataFrame, intended: np.ndarray) -> plt.Figure:
    P = coords[['x', 'y', 'z']].to_numpy(float)
    devs = deviations_along_path(P, intended)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(devs, bins=30, edgecolor='black')
    ax.set_xlabel('deviation (m)'); ax.set_ylabel('frequency'); ax.set_title('path deviation distribution')
    ax.grid(True)
    return fig


# ------------------------------- Public API -------------------------------

def generate_report_bundle(
    log_bytes: bytes,
    flightpath_bytes: Optional[bytes],
    workspace_dims: Optional[Dict[str, Tuple[float, float] | float]] = None
) -> Dict:
    """
    Main entry point for Dash.
    - Parses log + optional path from BYTES.
    - Trims to active flight if path provided.
    - Computes metrics.
    - Builds figures as base64 PNGs (no plt.show()).
    - Returns a bundle dict suitable for rendering UI.

    Returns dict with:
      report_md: str
      metrics: dict
      sections: list of {id,title,figs,metrics?}
      figures: {filename: base64_png}
      zip_bytes: bytes (optional)
    """
    # --- Parse log ---
    df, df_tel, df_qual = parse_log_bytes(log_bytes)
    coords = df.dropna(subset=['x', 'y', 'z']).reset_index(drop=True)

    if coords.empty:
        raise ValueError("No valid (x,y,z) rows found in the provided log")

    # --- Optional path ---
    intended = None
    if flightpath_bytes:
        intended = parse_flightpath_bytes(flightpath_bytes)
        if intended.size and intended.shape[1] >= 2:
            coords = trim_to_waypoints(coords, intended)

    # --- Time base ---
    t0 = coords['datetime'].min()
    coords['elapsed_s'] = (coords['datetime'] - t0).dt.total_seconds()

    # --- Dims for plotting ---
    inferred = infer_workspace_dims(coords)
    dims = compute_dims(workspace_dims, inferred)

    # --- Headline metrics placeholders ---
    path_metrics = {'mean': None, 'rms': None, 'pct95': None, 'max': None}
    within_bound = None
    wp_results = []
    wp_success_pct = None

    # --- Accuracy metrics (if path provided) ---
    if intended is not None and intended.size:
        # 2D path-accuracy metrics
        e_xy = deviations_along_path(coords[['x', 'y']].to_numpy(float), intended[:, :2])
        path_metrics = {
            'mean': float(np.mean(e_xy)),
            'rms': float(np.sqrt(np.mean(e_xy ** 2))),
            'max': float(np.max(e_xy)),
            'pct95': float(np.percentile(e_xy, 95)),
        }
        within_bound = bool((e_xy <= 0.10).mean() > 0.5)

        # Waypoint hit metrics
        P_xy = coords[['x', 'y']].to_numpy(float)
        hits = 0
        for i, wp in enumerate(intended):
            wp_xy = wp[:2]
            d_xy = np.linalg.norm(P_xy - wp_xy, axis=1)
            idx = int(np.argmin(d_xy))
            err = float(d_xy[idx])
            t = coords['datetime'].iloc[idx]
            hit = err <= 0.10
            hits += int(hit)
            wp_results.append({'index': i, 'error_m': err, 'time': t, 'hit': hit})
        if len(intended) > 0:
            wp_success_pct = 100.0 * hits / len(intended)

    # --- Build figures (base64) ---
    figures: Dict[str, str] = {}
    sections: List[Dict] = []

    def add_fig(name: str, fig_obj: Optional[plt.Figure]):
        if fig_obj is None:
            return
        figures[name] = _fig_to_b64(fig_obj)

    # Trajectory / time-series
    add_fig("3d_path.png", fig_path_3d(coords, dims, intended, use_elapsed=True))
    add_fig("xy.png",      fig_path_2d(coords, "xy", dims, intended, use_elapsed=True))
    add_fig("xz.png",      fig_path_2d(coords, "xz", dims, intended, use_elapsed=True))
    add_fig("yz.png",      fig_path_2d(coords, "yz", dims, intended, use_elapsed=True))
    add_fig("pos_time.png", fig_position_vs_time(coords))

    sections.append({
        "id": "trajectory",
        "title": "Trajectories",
        "figs": ["3d_path.png", "xy.png", "xz.png", "yz.png", "pos_time.png"]
    })

    # Latency / update rate
    lat_fig = fig_latency_hist(df)
    ur_fig  = fig_update_rate_hist(coords)
    figs_lr = []
    if lat_fig: add_fig("latency_hist.png", lat_fig); figs_lr.append("latency_hist.png")
    if ur_fig:  add_fig("update_rate_hist.png", ur_fig); figs_lr.append("update_rate_hist.png")
    if figs_lr:
        sections.append({"id": "latency", "title": "Latency & Update Rate", "figs": figs_lr})

    # Telemetry / quality
    figs_sig = []
    sig = fig_signal(df_tel)
    vol = fig_voltage(df_tel)
    qua = fig_quality(df_qual)
    if sig: add_fig("rssi.png", sig); figs_sig.append("rssi.png")
    if vol: add_fig("voltage.png", vol); figs_sig.append("voltage.png")
    if qua: add_fig("quality.png", qua); figs_sig.append("quality.png")
    if figs_sig:
        sections.append({"id": "signal", "title": "Signal & Quality", "figs": figs_sig})

    # Deviation metrics (only if path provided)
    if intended is not None and intended.size:
        add_fig("dev_time.png", fig_deviation_time(coords, intended))
        add_fig("dev_hist.png", fig_deviation_hist(coords, intended))
        sections.append({
            "id": "accuracy",
            "title": "Continuous Accuracy",
            "figs": ["dev_time.png", "dev_hist.png"],
            "metrics": ["path_mean", "path_rms", "path_p95"]
        })

    # --- Markdown report (no file writes) ---
    lines = []
    lines.append("# Marvelmind System Performance Report\n")
    if intended is not None and intended.size:
        lines.append("## Continuous Path-Following Accuracy")
        lines.append(f"- Mean error:      {path_metrics['mean']:.3f} m")
        lines.append(f"- RMS error:       {path_metrics['rms']:.3f} m")
        lines.append(f"- Max error:       {path_metrics['max']:.3f} m")
        lines.append(f"- 95th‐pct error:  {path_metrics['pct95']:.3f} m")
        if within_bound is not None:
            lines.append(f"- Within ±0.10 m:  {'✔' if within_bound else '✘'}")
        lines.append("")

        if wp_results:
            tol = 0.10
            wp_hits = sum(1 for r in wp_results if r['error_m'] <= tol)
            wp_total = len(wp_results)
            wp_success_pct_local = 100.0 * wp_hits / wp_total
            lines.append("## Waypoint Accuracy & Success Rate")
            lines.append(f"- Waypoints hit within {tol:.2f} m: {wp_success_pct_local:.1f}% ({wp_hits}/{wp_total})")
            lines.append("")
            lines.append("| WP # | Error (m) |   Time   | Hit? |")
            lines.append("|:----:|:---------:|:--------:|:----:|")
            for r in wp_results:
                idx = r['index']; err = r['error_m']; tstr = r['time'].strftime("%H:%M:%S"); hit = '✔' if r['hit'] else '✘'
                lines.append(f"| {idx:^4} | {err:^9.3f} | {tstr:^8} | {hit:^4} |")
            lines.append("")

    lines.append("## Position Stability (σ)")
    lines.append(f"- x: {coords['x'].std():.3f} m")
    lines.append(f"- y: {coords['y'].std():.3f} m")
    lines.append(f"- z: {coords['z'].std():.3f} m")
    lines.append("")

    total, avail = len(df), len(coords)
    lines.append("## Data Availability")
    if total > 0:
        lines.append(f"- {avail/total*100:.2f}% ({avail}/{total})")
    else:
        lines.append("- N/A")
    lines.append("")

    shifts = df.dropna(subset=['shift_ms'])['shift_ms']
    if not shifts.empty:
        lines.append("## Latency")
        lines.append(f"- avg: {shifts.mean():.1f} ms")
        lines.append(f"- σ: {shifts.std():.1f} ms")
        lines.append("")

    diffs = coords['datetime'].diff().dt.total_seconds().dropna()
    if not diffs.empty:
        update_rates = 1.0 / diffs
        lines.append("## Update Rate")
        lines.append(f"- avg: {update_rates.mean():.1f} Hz")
        lines.append(f"- σ: {update_rates.std():.1f} Hz")
        lines.append("")

    if not df_tel.empty or not df_qual.empty:
        lines.append("## Signal Quality")
        if not df_tel.empty:
            lines.append(f"- voltage: avg {df_tel['voltage_v'].mean():.2f} V, σ {df_tel['voltage_v'].std():.2f} V")
            lines.append(f"- rssi:    avg {df_tel['rssi_dbm'].mean():.1f} dBm, σ {df_tel['rssi_dbm'].std():.1f} dBm")
        if not df_qual.empty:
            lines.append(f"- location quality: avg {df_qual['quality_pct'].mean():.1f} %, σ {df_qual['quality_pct'].std():.1f} %")
        lines.append("")

    # coordinate resolution helper
    def _resolution(arr: pd.Series):
        vals = sorted(set(arr.dropna().to_numpy(dtype=float).tolist()))
        gaps = [j - i for i, j in zip(vals, vals[1:])]
        return min(gaps) if gaps else None

    lines.append("## Coordinate Resolution")
    rx, ry, rz = _resolution(coords['x']), _resolution(coords['y']), _resolution(coords['z'])
    lines.append(f"- x: {rx} m")
    lines.append(f"- y: {ry} m")
    lines.append(f"- z: {rz} m")
    lines.append("")

    if intended is not None and intended.size:
        # Compute 3D deviations for completeness in the markdown
        devs3d = deviations_along_path(coords[['x','y','z']].to_numpy(float), intended)
        lines.append("## Path Accuracy Metrics (3D)")
        lines.append(f"- mean deviation: {float(np.mean(devs3d)):.3f} m")
        lines.append(f"- max deviation:  {float(np.max(devs3d)):.3f} m")
        lines.append(f"- RMS deviation:  {float(np.sqrt(np.mean(devs3d**2))):.3f} m")
        lines.append(f"- 95th pct dev.:  {float(np.percentile(devs3d, 95)):.3f} m")
        lines.append("")

    report_md = "\n".join(lines)

    # --- Headline summary for UI ---
    metrics_headline = {
        "path_mean": path_metrics['mean'],
        "path_rms": path_metrics['rms'],
        "path_p95": path_metrics['pct95'],
        "path_max": path_metrics['max'],
        "wp_success_pct": wp_success_pct,
        "availability_pct": (avail/total*100.0) if total > 0 else None,
    }

    # --- Optional ZIP bundle (report + images) ---
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("report.md", report_md.encode("utf-8"))
        for fname, b64 in figures.items():
            zf.writestr(fname, base64.b64decode(b64))
    zip_buf.seek(0)
    zip_bytes = zip_buf.read()
    zip_b64 = base64.b64encode(zip_bytes).decode("ascii")  # <-- JSON-safe

    # --- Build and return JSON-safe bundle (no raw bytes) ---
    bundle = {
        "report_md": report_md,
        "metrics": metrics_headline,
        "sections": sections,
        "figures": figures,   # filename -> base64 PNG (string)
        "zip_b64": zip_b64,   # <-- use this instead of zip_bytes
        # Optionally add metadata for convenience:
        # "zip_filename": "report_bundle.zip",
        # "zip_mime": "application/zip",
    }
    return bundle


# ------------------------------- Local test (optional) -------------------------------
if __name__ == "__main__":
    # Simple sanity check for local use (reads from disk just for testing)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to Marvelmind log CSV")
    ap.add_argument("--path", help="Path to intended flightpath CSV/DPT", default=None)
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--W", type=float, default=None)
    ap.add_argument("--H", type=float, default=None)
    args = ap.parse_args()

    with open(args.log, "rb") as f:
        log_b = f.read()
    fp_b = None
    if args.path:
        with open(args.path, "rb") as f:
            fp_b = f.read()

    dims = None
    if args.L or args.W or args.H:
        dims = {
            "x": (0.0, args.L) if args.L else None,
            "y": (0.0, args.W) if args.W else None,
            "z": (0.0, args.H) if args.H else None,
        }

    bundle = make_json_safe(generate_report_bundle(log_b, fp_b, dims))
    print("Bundle keys:", bundle.keys())
    print("Sections:", [s["title"] for s in bundle["sections"]])
    print("Metrics:", bundle["metrics"])
    print("Figures:", list(bundle["figures"].keys()))
    # Write the zip for quick preview
    with open("report_bundle.zip", "wb") as f:
        f.write(base64.b64decode(bundle["zip_b64"]))
    print("Wrote report_bundle.zip")
