#!/usr/bin/env python3
"""
Simple network detection on prebuilt multi-station MiniSEED windows.

- Uses flovopy.processing.detection.detect_network_event()
- LAUNCH pass only by default (SONIC optional via --enable-sonic)
- ONE MiniSEED per detected event (whole Stream snippet)
- Outputs a CSV of detections
- Snippets organized as <mseed-out>/<YYYY>/<MM>/...

Examples:
  python detect_simple.py \
    --mseed-root /data/KSC/all_florida_launches/processed_mseed \
    --out /data/KSC/detections.csv \
    --mseed-out /data/KSC/event_snippets

  python detect_simple.py \
    --csv-with-mseed windows.csv \
    --out det.csv \
    --mseed-out event_snips
"""

import os, glob, argparse, warnings
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd
from obspy import read, UTCDateTime, Stream

from flovopy.processing.detection import detect_network_event  # <<— swap-in
from flovopy.seisanio.utils.helpers import write_wavfile

# ------------------------------ Helpers ------------------------------

def _merge_intervals(intervals, guard: float):
    """Merge overlapping [t1-guard, t2+guard] windows."""
    if not intervals:
        return []
    ext = [(t1 - guard, t2 + guard) for (t1, t2) in intervals]
    ext.sort(key=lambda x: x[0])
    merged = [list(ext[0])]
    for s, e in ext[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]

def _mask_inplace(st: Stream, intervals: List[Tuple[UTCDateTime, UTCDateTime]]):
    """Zero out samples in [t1, t2] for each trace (exclusive labeling)."""
    if not intervals:
        return
    for tr in st:
        sr = float(tr.stats.sampling_rate or 0.0)
        if sr <= 0:
            continue
        for t1, t2 in intervals:
            i1 = max(0, int((t1 - tr.stats.starttime) * sr))
            i2 = min(tr.stats.npts, int((t2 - tr.stats.starttime) * sr))
            if i2 > i1:
                tr.data[i1:i2] = 0

def _year_month_dir(root: str, t_on: UTCDateTime, enable: bool) -> str:
    if not enable:
        os.makedirs(root, exist_ok=True)
        return root
    yyyy = f"{t_on.year:04d}"
    mm   = f"{t_on.month:02d}"
    out_subdir = os.path.join(root, yyyy, mm)
    os.makedirs(out_subdir, exist_ok=True)
    return out_subdir


def _write_event_stream(
    st_full: Stream,
    t_on: UTCDateTime,
    t_off: UTCDateTime,
    out_root: str,
    *,
    pre_pad: float = 10.0,
    post_pad: float = 10.0,
    year_month_dirs: bool = True,
    dbstring: str = "DET__",  # <- choose your DB tag (e.g., KSC / MONTSERRAT / etc.)
) -> str:
    """
    Trim WHOLE stream to [t_on - pre_pad, t_off + post_pad] and write ONE MiniSEED
    using flovopy.seisanio.helpers.write_wavfile(), which creates:
        <out>/<YYYY>/<MM>/<YYYY>-<MM>-<DD>-<HHMM>-<SS>S.<DBSTRING>_<NCH>.mseed
    """
    t1 = t_on - pre_pad
    t2 = t_off + post_pad

    st_cut = st_full.copy().trim(t1, t2, pad=False)
    if len(st_cut) == 0:
        raise RuntimeError("Empty stream after trim")

    # write_wavfile() handles basename + year/month subdirs
    out_path = write_wavfile(
        st=st_cut,
        out_root=out_root,
        dbstring=dbstring,
        numchans=len(st_cut),
        year_month_dirs=year_month_dirs,
        fmt="MSEED",
    )
    return out_path

def _run_detect_with_flovo(
    st: Stream,
    *,
    freq_band: Optional[Tuple[float, float]],
    sta: float,
    lta: float,
    on: float,
    off: float,
    minchans: int,
    pad: float = 0.0,
    algorithm: str = "recstalta",
    criterion: str = "longest",
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Wrapper over flovopy.processing.detection.detect_network_event -> normalized list."""
    freq = list(freq_band) if (freq_band and len(freq_band) == 2) else None
    trig = []
    while True:
        trig, ontimes, offtimes = detect_network_event(
            st_in=st,
            minchans=minchans,
            threshon=on,
            threshoff=off,
            sta=sta,
            lta=lta,
            pad=pad,
            best_only=False,
            verbose=verbose,
            freq=freq,
            algorithm=algorithm,
            criterion=criterion,
            join_within=lta,
            min_duration=lta,
        )
        if trig:
            break
        sta=sta*0.75
        lta=lta*0.75
        on=(3*on+1.0)/4
        off=(3*off+1.0)/4
        if on<2.0:
            break
        

    events: List[Dict[str, Any]] = []
    if trig and ontimes and offtimes:
        for ev, t_on, t_off in zip(trig, ontimes, offtimes):
            t_on = UTCDateTime(t_on)
            t_off = UTCDateTime(t_off)
            events.append({"t_on": t_on, "t_off": t_off, "meta": ev})
        events.sort(key=lambda r: r["t_on"])
    return events


# ------------------------------ CLI / main ------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv-with-mseed", help="CSV containing 'mseed_path' column.")
    src.add_argument("--mseed-root", help="Directory to recursively scan for *.mseed")

    ap.add_argument("--out", required=True, help="Output CSV for detections")
    ap.add_argument("--mseed-out", required=True, help="Directory to write event MiniSEED snippets (one per event)")
    ap.add_argument("--year-month-dirs", action="store_true", default=True,
                    help="Write output snippets into out/YYYY/MM/ (default: on).")

    ap.add_argument("--enable-sonic", action="store_true",
                    help="Run the SONIC pass too (default: off).")

    ap.add_argument("--minchans", type=int, default=2)
    ap.add_argument("--mask-guard", type=float, default=5.0, help="Seconds to extend launch intervals before masking for SONIC.")
    ap.add_argument("--prepad", type=float, default=10.0, help="Pre-trigger seconds for snippets.")
    ap.add_argument("--postpad", type=float, default=10.0, help="Post-trigger seconds for snippets.")

    # Launch pass params (mapped to detect_network_event)
    ap.add_argument("--launch-band", type=float, nargs=2, metavar=("F1","F2"), default=[0.5, 12.0])
    ap.add_argument("--launch-sta", type=float, default=5.0)
    ap.add_argument("--launch-lta", type=float, default=60.0)
    ap.add_argument("--launch-on", type=float, default=3.5)
    ap.add_argument("--launch-off", type=float, default=1.2)
    ap.add_argument("--launch-pad", type=float, default=0.0)
    ap.add_argument("--launch-criterion", default="longest",
                    choices=("longest","cft","cft_duration"))

    # Sonic pass params (disabled by default)
    ap.add_argument("--sonic-band", type=float, nargs=2, metavar=("F1","F2"), default=[1.0, 20.0])
    ap.add_argument("--sonic-sta", type=float, default=0.25)
    ap.add_argument("--sonic-lta", type=float, default=5.0)
    ap.add_argument("--sonic-on", type=float, default=6.0)
    ap.add_argument("--sonic-off", type=float, default=2.0)
    ap.add_argument("--sonic-pad", type=float, default=0.0)
    ap.add_argument("--sonic-criterion", default="longest",
                    choices=("longest","cft","cft_duration"))

    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(args.mseed_out, exist_ok=True)

    # Collect inputs
    if args.csv_with_mseed:
        df = pd.read_csv(args.csv_with_mseed)
        if "mseed_path" not in df.columns:
            raise ValueError("CSV must include 'mseed_path'.")
        inputs = [str(p) for p in df["mseed_path"].tolist() if isinstance(p, str) and len(p)]
    else:
        inputs = sorted(glob.glob(os.path.join(args.mseed_root, "**", "*.mseed"), recursive=True))

    det_rows: List[Dict[str, Any]] = []

    for idx, path in enumerate(inputs):
        try:
            st_full = read(path)
        except Exception as e:
            warnings.warn(f"Failed to read {path}: {e}")
            continue

        # ---------- LAUNCH (via detect_network_event) ----------
        launch_events = _run_detect_with_flovo(
            st_full,
            freq_band=tuple(args.launch_band),
            sta=args.launch_sta,
            lta=args.launch_lta,
            on=args.launch_on,
            off=args.launch_off,
            minchans=max(args.minchans, 3),
            pad=float(args.launch_pad),
            algorithm="recstalta",
            criterion=args.launch_criterion,
        )

        launch_intervals = [(ev["t_on"], ev["t_off"]) for ev in launch_events]

        # Write launch snippets (one per event)
        for k, ev in enumerate(launch_events, 1):
            try:
                outp = _write_event_stream(
                    st_full, ev["t_on"], ev["t_off"],
                    out_root=args.mseed_out,
                    pre_pad=args.prepad,
                    post_pad=args.postpad,
                    year_month_dirs=bool(args.year_month_dirs),
                    dbstring="LAUNCH",  # or whatever database tag you want
                )
            except Exception as e:
                warnings.warn(f"Write failed (launch) for {path}: {e}")
                continue

            det_rows.append({
                "input_mseed": path,
                "phase": "launch",
                "event_index_in_file": k,
                "onset_utc": str(ev["t_on"]),
                "offset_utc": str(ev["t_off"]),
                "duration_s": float(ev["t_off"] - ev["t_on"]),
                "coincidence_sum": ev["meta"].get("coincidence_sum"),
                "cft_peak_wmean": ev["meta"].get("cft_peak_wmean"),
                "out_mseed": outp,
            })

        # ---------- SONIC (optional; disabled by default) ----------
        if args.enable_sonic:
            sonic_source = st_full.copy()
            if launch_intervals:
                merged = _merge_intervals(launch_intervals, guard=float(args.mask_guard))
                _mask_inplace(sonic_source, merged)

            sonic_events = _run_detect_with_flovo(
                sonic_source,
                freq_band=tuple(args.sonic_band),
                sta=args.sonic_sta,
                lta=args.sonic_lta,
                on=args.sonic_on,
                off=args.sonic_off,
                minchans=max(args.minchans, 2),
                pad=float(args.sonic_pad),
                algorithm="recstalta",
                criterion=args.sonic_criterion,
            )

            for k, ev in enumerate(sonic_events, 1):
                try:
                    outp = _write_event_stream(
                        st_full, ev["t_on"], ev["t_off"],
                        out_root=args.mseed_out,
                        pre_pad=args.prepad,
                        post_pad=args.postpad,
                        year_month_dirs=bool(args.year_month_dirs),
                        dbstring="SONIC",  # or whatever database tag you want
                    )
                except Exception as e:
                    warnings.warn(f"Write failed (sonic) for {path}: {e}")
                    continue

                det_rows.append({
                    "input_mseed": path,
                    "phase": "sonic",
                    "event_index_in_file": k,
                    "onset_utc": str(ev["t_on"]),
                    "offset_utc": str(ev["t_off"]),
                    "duration_s": float(ev["t_off"] - ev["t_on"]),
                    "coincidence_sum": ev["meta"].get("coincidence_sum"),
                    "cft_peak_wmean": ev["meta"].get("cft_peak_wmean"),
                    "out_mseed": outp,
                })

    # Write detection table
    pd.DataFrame(det_rows).to_csv(args.out, index=False)
    print(f"Done. Wrote {len(det_rows)} detections to {args.out}")
    print(f"Event MiniSEED snippets in: {args.mseed_out} (organized by YYYY/MM)")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        sys.argv += [
            "--mseed-root", "/data/KSC/all_florida_launches/preprocessed_mseed",
            "--out", "/data/KSC/all_florida_launches/detections_simple2.csv",
            "--mseed-out", "/data/KSC/all_florida_launches/network_detected",
            "--minchans", "3",
            "--mask-guard", "5",
            # SONIC remains OFF unless explicitly enabled with --enable-sonic
        ]
    main()