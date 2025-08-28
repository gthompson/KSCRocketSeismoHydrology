#!/usr/bin/env python3
"""
launch_booster_detection_v2.py

Compact pipeline:
  1) Read event windows CSV
  2) For each event:
     - LAUNCH pass (long-duration tuning) using sds2event preset + network detector
     - (Optional) BOOSTER pass (short-duration tuning) in a delayed window after launch
     - (Optional) SONICBAT mode: run short detector in the primary window too
  3) Emit CSV rows and (optional) MiniSEED snippets + quicklooks

Relies on:
- flovopy.sds.sds2event.load_event_stream_from_sds
- flovopy.processing.detection.detect_network_event (network/association)
- flovopy.processing.detection.add_sta_lta_triggers_to_stream (per-trace overlays)
- flovopy.core.miniseed_io.write_mseed
"""

import os
import sys
import argparse
import warnings
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from obspy import UTCDateTime

from flovopy.sds.sds import SDSobj
from flovopy.sds.sds2event import load_event_stream_from_sds
from flovopy.processing.detection import (
    detect_network_event,
    add_sta_lta_triggers_to_stream,
)
from flovopy.processing.metrics import estimate_snr
from flovopy.core.miniseed_io import write_mseed

# Optional quicklooks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------ Helpers ------------------------------

def to_utc(val) -> UTCDateTime:
    """Robust convert of epoch/ISO(±TZ)/'...Z' into UTCDateTime."""
    if val is None or (isinstance(val, float) and np.isnan(val)) or pd.isna(val):
        raise ValueError("Missing time value")
    if isinstance(val, (int, float, np.integer, np.floating)):
        return UTCDateTime(float(val))
    s = str(val).strip()
    if "T" not in s and " " in s:
        s = s.replace(" ", "T", 1)
    ts = pd.to_datetime(s.replace("Z", "+00:00"), utc=True, errors="coerce")
    if not pd.isna(ts):
        return UTCDateTime(ts.to_pydatetime())
    return UTCDateTime(s)

def load_events(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    cols = set(df.columns)
    if {"start_time", "end_time"} <= cols:
        df["t_start"] = df["start_time"].apply(to_utc)
        df["t_end"] = df["end_time"].apply(to_utc)
    elif {"window_start", "window_end"} <= cols:
        df["t_start"] = df["window_start"].apply(to_utc)
        df["t_end"] = df["window_end"].apply(to_utc)
    elif {"t0", "window_seconds"} <= cols:
        df["t0"] = df["t0"].apply(to_utc)
        df["t_start"] = df["t0"]
        df["t_end"] = df["t0"] + df["window_seconds"].astype(float)
    else:
        raise ValueError(
            "CSV must have {start_time,end_time} or {window_start,window_end} or {t0,window_seconds}"
        )
    return df

def looks_like_spacex(ev_meta: Dict[str, Any]) -> bool:
    keys = ("rocket", "mission", "provider", "launch_provider")
    text = " ".join(str(ev_meta.get(k, "")).lower() for k in keys)
    return ("spacex" in text) or ("falcon" in text)

def collect_event_meta(row: pd.Series) -> Tuple[Dict[str, Any], Optional[str]]:
    reserved = {"t_start", "t_end", "net", "sta", "loc", "cha",
                "window_start", "window_end", "start_time", "end_time"}
    meta = {}
    for col in row.index:
        if col in {"t_start", "t_end"}:
            continue
        key = f"event_{col}" if col in reserved else col
        meta[key] = row[col]
    eid = meta.get("event_event_id") or meta.get("event_id") or None
    return meta, eid

def _snr_ok_any_channel(st, t_on: UTCDateTime, t_off: UTCDateTime, snr_min: float) -> bool:
    """True if at least one channel meets SNR >= snr_min using time-domain method='std'."""
    try:
        # Evaluate SNR per-trace and accept if any are above threshold
        for tr in st:
            snr_list, _ = estimate_snr(tr.copy(), method="std", split_time=(t_on, t_off))
            if snr_list and any(np.isfinite(s) and s >= snr_min for s in snr_list):
                return True
        return False
    except Exception:
        return False

def safe_loc(loc: str) -> str:
    return loc if (loc and str(loc).strip()) else "--"

def make_event_filename(base_dir: str, event_idx: int, net: str, sta: str, loc: str, cha: str,
                        t_on: UTCDateTime, rank: int, phase: str,
                        event_id: Optional[str] = None, cluster_id: Optional[int] = None) -> str:
    loc_str = safe_loc(loc)
    tstr = t_on.format_iris_web_service().replace(":", "-").replace(".", "-")
    prefix = f"{event_idx:06d}"
    if event_id:
        safe_eid = "".join(c for c in str(event_id) if c.isalnum() or c in ("-", "_"))
        prefix = f"{safe_eid}__{prefix}"
    phase_code = "L" if phase == "launch" else "B"
    cl_tag = f"__CL{cluster_id:03d}" if cluster_id is not None else ""
    fname = f"{prefix}__PHASE-{phase_code}{cl_tag}__{tstr}__{net}.{sta}.{loc_str}.{cha}__r{rank}.mseed"
    return os.path.join(base_dir, net, sta, loc_str, cha, fname)

def write_event_snippet(tr, t_on: Optional[UTCDateTime], t_off: Optional[UTCDateTime],
                        rank: int, mseed_dir: Optional[str], event_idx: int, phase: str,
                        event_id: Optional[str] = None, cluster_id: Optional[int] = None,
                        pad_before: float = 5.0, pad_after: float = 15.0, fallback_duration: float = 60.0) -> Optional[str]:
    if not mseed_dir or t_on is None:
        return None
    try:
        t1 = max(tr.stats.starttime, (t_on - pad_before) if t_on else tr.stats.starttime)
        t2 = min(tr.stats.endtime,
                 (t_off + pad_after) if t_off is not None else ((t_on + fallback_duration) if t_on else tr.stats.endtime))
        if t2 <= t1:
            return None
        tr_cut = tr.copy().trim(t1, t2, pad=False)
        net, sta, loc, cha = tr_cut.id.split(".")
        outpath = make_event_filename(mseed_dir, event_idx, net, sta, loc, cha, t_on, rank, phase,
                                      event_id=event_id, cluster_id=cluster_id)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        write_mseed(tr_cut, outpath)
        return outpath
    except Exception as e:
        warnings.warn(f"MiniSEED write failed for {tr.id}: {e}")
        return None

def save_quicklook(tr, picks: List[Dict[str, UTCDateTime]], out_png: str):
    try:
        sr = float(tr.stats.sampling_rate or 0.0)
        if sr <= 0: return
        t_rel = np.arange(tr.stats.npts) / sr
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_rel, tr.data, lw=0.8)
        for k, p in enumerate(picks, 1):
            ton_rel = (p["t_on"] - tr.stats.starttime)
            ax.axvline(ton_rel, ls="--", lw=1.0, label=f"Trig {k}")
        ax.set_xlabel("Time (s since window start)")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")
        ax.set_title(f"{tr.id}  {tr.stats.starttime}–{tr.stats.endtime}")
        fig.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception as e:
        warnings.warn(f"Quicklook fail for {tr.id}: {e}")


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv", required=True, help="CSV of event windows")
    ap.add_argument("--sds-root", required=True, help="Root of SDS archive")

    # Primary window padding
    ap.add_argument("--pad-before", type=float, default=60.0)
    ap.add_argument("--pad-after", type=float, default=600.0)

    # STA/LTA for LAUNCH
    ap.add_argument("--sta", type=float, default=1.0)
    ap.add_argument("--lta", type=float, default=20.0)
    ap.add_argument("--on", type=float, default=3.5)
    ap.add_argument("--off", type=float, default=1.5)
    ap.add_argument("--max-two-triggers", action="store_true")
    ap.add_argument("--min-sep", type=float, default=45.0)

    # Association & constraints
    ap.add_argument("--assoc-min-stations", type=int, default=2)
    ap.add_argument("--launch-min-duration", type=float, default=20.0)
    ap.add_argument("--launch-max-duration", type=float, default=300.0)
    ap.add_argument("--launch-snr-min", type=float, default=5.0)

    # Booster / SonicBAT tuning
    ap.add_argument("--booster-search", action="store_true", default=True)
    ap.add_argument("--booster-on-all", action="store_true")
    ap.add_argument("--booster-min-delay", type=float, default=300.0)
    ap.add_argument("--booster-max-delay", type=float, default=900.0)
    ap.add_argument("--booster-sta", type=float, default=0.25)
    ap.add_argument("--booster-lta", type=float, default=5.0)
    ap.add_argument("--booster-on", type=float, default=4.5)
    ap.add_argument("--booster-off", type=float, default=2.0)
    ap.add_argument("--booster-max-trigs", type=int, default=3)
    ap.add_argument("--booster-min-duration", type=float, default=0.3)
    ap.add_argument("--booster-max-duration", type=float, default=2.5)
    ap.add_argument("--booster-snr-min", type=float, default=5.0)

    # Output
    ap.add_argument("--quicklooks", default=None, help="Directory for PNG quicklooks (optional)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--mseed-out", default=None, help="Directory to write MiniSEED snippets (optional)")
    ap.add_argument("--event-pad-before", type=float, default=5.0)
    ap.add_argument("--event-pad-after", type=float, default=15.0)
    ap.add_argument("--fallback-duration", type=float, default=60.0)

    # SonicBAT generic mode
    ap.add_argument("--sonicbat-mode", action="store_true",
                    help="Also run short detector in primary window (aircraft datasets, etc.)")

    # Fast test defaults
    ap.add_argument("--use-defaults", action="store_true",
                    help="Ignore CLI args and use embedded defaults (for quick tests)")

    args = ap.parse_args()

    #if args.use-defaults and len(sys.argv) > 1:
    #    print("[INFO] --use-defaults: using internal default paths and parameters.")

    df = load_events(args.csv)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.quicklooks:
        os.makedirs(args.quicklooks, exist_ok=True)
    if args.mseed_out:
        os.makedirs(args.mseed_out, exist_ok=True)

    sdsin = SDSobj(args.sds_root)
    rows: List[Dict[str, Any]] = []

    for idx, ev in df.iterrows():
        ev_meta, this_event_id = collect_event_meta(ev)
        t1_launch = ev["t_start"] - args.pad_before
        t2_launch = ev["t_end"] + args.pad_after

        # -------- LAUNCH (long) --------
        # launch window: gap-normalize only (do STA/LTA on raw-ish data)
        st_launch = load_event_stream_from_sds(
            sds_root=args.sds_root,
            t1=t1_launch,
            t2=t2_launch,
            preset="archive_preset",   # or "analysis_preset" if you want filtering here
            inv=None,                  # pass Inventory only if using "analysis_preset"
            verbose=False,
            speed=1,
        )
        if len(st_launch) == 0:
            continue

        # Network/coincidence detection (association included)
        trigs, _, _ = detect_network_event(
            st_launch,
            minchans=max(args.assoc_min_stations, 2),
            threshon=args.on, threshoff=args.off,
            sta=args.sta, lta=args.lta,
            pad=0.0, best_only=False, verbose=False,
            freq=None, algorithm="recstalta", criterion="cft",
        )

        launch_events = []
        for e in (trigs or []):
            t_on = UTCDateTime(e["time"])
            t_off = t_on + e["duration"]
            dur_ok = args.launch_min_duration <= float(t_off - t_on) <= args.launch_max_duration
            snr_ok = True if args.launch_snr_min <= 0 else _snr_ok_any_channel(st_launch, t_on, t_off, args.launch_snr_min)
            if dur_ok and snr_ok:
                launch_events.append({"t_on": t_on, "t_off": t_off, "meta": e})

        launch_events.sort(key=lambda r: r["t_on"])
        chosen_launch = launch_events[0] if launch_events else None

        # Persist per-trace triggers for overlays (optional)
        add_sta_lta_triggers_to_stream(
            st_launch,
            args.sta, args.lta, args.on, args.off,
            max_trigs=(2 if args.max_two_triggers else 1),
            min_sep_s=args.min_sep,
            preprocess=False, save_to_stats=True,
        )

        # Emit rows/files for all launch clusters (rank by CFT around onset)
        for cl_id, rec in enumerate(launch_events):
            t_on, t_off, meta = rec["t_on"], rec["t_off"], rec["meta"]
            # rank members by descending CFT peak; use provided trace_ids
            trace_ids = meta.get("trace_ids", [])
            # Safe fallback: if trace_ids missing, rank by channel order
            ranked_ids = trace_ids if trace_ids else [tr.id for tr in st_launch]
            for rank, tid in enumerate(ranked_ids, 1):
                tr = st_launch.select(id=tid)[0] if st_launch.select(id=tid) else st_launch[0]
                mseed_path = write_event_snippet(
                    tr, t_on, t_off, rank=rank, mseed_dir=args.mseed_out,
                    event_idx=int(idx), phase="launch", event_id=this_event_id,
                    cluster_id=cl_id,
                    pad_before=args.event_pad_before, pad_after=args.event_pad_after,
                    fallback_duration=args.fallback_duration,
                )
                if args.quicklooks and t_on:
                    ql = os.path.join(
                        args.quicklooks,
                        f"{int(idx):06d}_LAUNCH_CL{cl_id:03d}_{tr.id.replace('.', '_')}_r{rank}.png"
                    )
                    save_quicklook(tr, [{"t_on": t_on}], ql)

                rows.append({
                    **ev_meta, "phase": "launch",
                    "event_index": int(idx),
                    "net": tr.stats.network, "sta": tr.stats.station,
                    "loc": tr.stats.location, "cha": tr.stats.channel,
                    "window_start": str(t1_launch), "window_end": str(t2_launch),
                    "trigger_rank": rank,
                    "trigger_on_time": str(t_on), "trigger_off_time": str(t_off),
                    "duration_s": float(t_off - t_on),
                    "coincidence_sum": meta.get("coincidence_sum"),
                    "cft_peak_wmean": meta.get("cft_peak_wmean"),
                    "cluster_id": cl_id,
                    "is_launch_choice": bool(chosen_launch and t_on == chosen_launch["t_on"]),
                    "mseed_path": mseed_path,
                })

        # -------- BOOSTER (short) --------
        t_launch_on = chosen_launch["t_on"] if chosen_launch else (
            min([UTCDateTime(e["time"]) for e in (trigs or [])], default=None)
        )

        do_booster = bool(args.booster_search) and (args.booster_on_all or looks_like_spacex(ev_meta))
        if do_booster and t_launch_on is not None:
            tb1 = t_launch_on + args.booster_min_delay
            tb2 = t_launch_on + args.booster_max_delay

            st_boost = load_event_stream_from_sds(
                sds_root=args.sds_root,
                t1=tb1,
                t2=tb2,
                preset="archive_preset",   # keep raw-ish for short-impulse search
                verbose=False,
            )
            if len(st_boost):
                trigs_b, _, _ = detect_network_event(
                    st_boost,
                    minchans=max(args.assoc_min_stations, 2),
                    threshon=args.booster_on, threshoff=args.booster_off,
                    sta=args.booster_sta, lta=args.booster_lta,
                    pad=0.0, best_only=False, verbose=False,
                    freq=None, algorithm="recstalta", criterion="cft",
                )
                boost_events = []
                for e in (trigs_b or []):
                    t_on = UTCDateTime(e["time"])
                    t_off = t_on + e["duration"]
                    dur_s = float(t_off - t_on)
                    if not (args.booster_min_duration <= dur_s <= args.booster_max_duration):
                        continue
                    # SNR gate (looser here, but still usable)
                    snr_ok = True if args.booster_snr_min <= 0 else _snr_ok_any_channel(st_boost, t_on, t_off, args.booster_snr_min)
                    if snr_ok:
                        boost_events.append({"t_on": t_on, "t_off": t_off, "meta": e})

                boost_events.sort(key=lambda r: r["t_on"])
                for cl_id, rec in enumerate(boost_events):
                    t_on, t_off, meta = rec["t_on"], rec["t_off"], rec["meta"]
                    ranked_ids = meta.get("trace_ids") or [tr.id for tr in st_boost]
                    for rank, tid in enumerate(ranked_ids, 1):
                        tr = st_boost.select(id=tid)[0] if st_boost.select(id=tid) else st_boost[0]
                        mseed_path = write_event_snippet(
                            tr, t_on, t_off, rank=rank, mseed_dir=args.mseed_out,
                            event_idx=int(idx), phase="booster", event_id=this_event_id,
                            cluster_id=cl_id,
                            pad_before=args.event_pad_before, pad_after=args.event_pad_after,
                            fallback_duration=args.fallback_duration,
                        )
                        if args.quicklooks and t_on:
                            ql = os.path.join(
                                args.quicklooks,
                                f"{int(idx):06d}_BOOST_CL{cl_id:03d}_{tr.id.replace('.', '_')}_r{rank}.png"
                            )
                            save_quicklook(tr, [{"t_on": t_on}], ql)

                        rows.append({
                            **ev_meta, "phase": "booster",
                            "event_index": int(idx),
                            "net": tr.stats.network, "sta": tr.stats.station,
                            "loc": tr.stats.location, "cha": tr.stats.channel,
                            "window_start": str(tb1), "window_end": str(tb2),
                            "trigger_rank": rank,
                            "trigger_on_time": str(t_on), "trigger_off_time": str(t_off),
                            "duration_s": float(t_off - t_on),
                            "coincidence_sum": meta.get("coincidence_sum"),
                            "cft_peak_wmean": meta.get("cft_peak_wmean"),
                            "cluster_id": cl_id,
                            "mseed_path": mseed_path,
                        })

        # -------- SONICBAT on primary window (optional) --------
        if args.sonicbat_mode:
            # Just rerun the short detector on the primary window stream:
            st_bat = load_event_stream_from_sds(
                sdsin, t1_launch, t2_launch,
                preset="detect_short",
                preprocess_overrides={"long_gap_fill": "zero"},
                read_kwargs={"speed": 1},
            )
            if len(st_bat):
                trigs_sb, _, _ = detect_network_event(
                    st_bat,
                    minchans=max(args.assoc_min_stations, 2),
                    threshon=args.booster_on, threshoff=args.booster_off,
                    sta=args.booster_sta, lta=args.booster_lta,
                    pad=0.0, best_only=False, verbose=False,
                    freq=None, algorithm="recstalta", criterion="cft",
                )
                sb_events = []
                for e in (trigs_sb or []):
                    t_on = UTCDateTime(e["time"])
                    t_off = t_on + e["duration"]
                    dur_s = float(t_off - t_on)
                    if not (args.booster_min_duration <= dur_s <= args.booster_max_duration):
                        continue
                    if args.booster_snr_min <= 0 or _snr_ok_any_channel(st_bat, t_on, t_off, args.booster_snr_min):
                        sb_events.append({"t_on": t_on, "t_off": t_off, "meta": e})

                sb_events.sort(key=lambda r: r["t_on"])
                for cl_id, rec in enumerate(sb_events):
                    t_on, t_off, meta = rec["t_on"], rec["t_off"], rec["meta"]
                    ranked_ids = meta.get("trace_ids") or [tr.id for tr in st_bat]
                    for rank, tid in enumerate(ranked_ids, 1):
                        tr = st_bat.select(id=tid)[0] if st_bat.select(id=tid) else st_bat[0]
                        mseed_path = write_event_snippet(
                            tr, t_on, t_off, rank=rank, mseed_dir=args.mseed_out,
                            event_idx=int(idx), phase="sonicbat", event_id=this_event_id,
                            cluster_id=cl_id,
                            pad_before=args.event_pad_before, pad_after=args.event_pad_after,
                            fallback_duration=args.fallback_duration,
                        )
                        rows.append({
                            **ev_meta, "phase": "sonicbat",
                            "event_index": int(idx),
                            "net": tr.stats.network, "sta": tr.stats.station,
                            "loc": tr.stats.location, "cha": tr.stats.channel,
                            "window_start": str(t1_launch), "window_end": str(t2_launch),
                            "trigger_rank": rank,
                            "trigger_on_time": str(t_on), "trigger_off_time": str(t_off),
                            "duration_s": float(t_off - t_on),
                            "coincidence_sum": meta.get("coincidence_sum"),
                            "cft_peak_wmean": meta.get("cft_peak_wmean"),
                            "cluster_id": cl_id,
                            "mseed_path": mseed_path,
                        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote results to {args.out} (rows={len(out_df)})")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        defaults = [
            "--csv", os.path.join(os.path.dirname(__file__), "all_florida_launches.csv"),
            "--sds-root", "/data/remastered/SDS_KSC",
            "--out", os.path.join(os.path.dirname(__file__), "results_sta_lta_v2.csv"),
            "--mseed-out", "/data/KSC/all_florida_launches",
            "--quicklooks", "/data/KSC/all_florida_launches",
            "--pad-before", "60", "--pad-after", "600",
            "--sta", "1.0", "--lta", "20.0", "--on", "3.5", "--off", "1.5",
            "--max-two-triggers", "--min-sep", "45",
            "--assoc-min-stations", "2",
            "--launch-min-duration", "20", "--launch-max-duration", "300", "--launch-snr-min", "5",
            "--booster-min-delay", "300", "--booster-max-delay", "900",
            "--booster-sta", "0.25", "--booster-lta", "5.0", "--booster-on", "4.5", "--booster-off", "2.0",
            "--booster-max-trigs", "3", "--booster-min-duration", "0.3", "--booster-max-duration", "2.5",
            "--booster-snr-min", "5",
        ]
        sys.argv.extend(defaults)
    main()