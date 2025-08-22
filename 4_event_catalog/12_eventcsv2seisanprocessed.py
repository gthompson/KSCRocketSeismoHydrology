import os
from flovopy.sds.sds2event import csv_to_event_miniseed
csvfile = os.path.join(os.path.dirname(__file__), "all_florida_launches.csv")

summary = csv_to_event_miniseed(
    sds_root="/data/remastered/SDS_KSC",
    csv_path=csvfile,
    start_col="window_start",
    end_col="window_end",            # or None if your CSV has only starts
    out_dir="/data/KSC/all_florida_launches/processed_mseed",
    pad_before=120.0,
    pad_after=600.0,
    net="*", sta="*", loc="*", cha="*",
    preset="archive_preset",       # preserves raw spectrum
    year_month_dirs=True,          # default; SEISAN-like /YYYY/MM
    event_id_col="event_id",       # optional
    verbose=False,
)
print(summary.head())