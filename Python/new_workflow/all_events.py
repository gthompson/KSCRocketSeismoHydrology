import pandas as pd
from obspy import UTCDateTime
from obspy.clients.filesystem.sds import Client

# Path to SDS archive
sds_root = '/data/SDS_SAFE'
client = Client(sds_root)

# Load CSV file of rocket launches
csv_path = '/home/thompsong/Developer/KSCRocketSeismoHydrology/Python/new_workflow/all_florida_launches.csv'
output_path = csv_path.replace('.csv', '_with_seed_ids.csv')

# Load CSV with datetime parsing
df = pd.read_csv(csv_path, parse_dates=['window_start', 'window_end'])

# Create column to hold SEED IDs
df['SEED_ids'] = None

valid_networks = {'AM', '1R', 'XA', 'FL'}

for index, row in df.iterrows():
    try:
        stime = UTCDateTime(row['window_start'])
        etime = UTCDateTime(row['window_end'])

        if (etime - stime) > 8 * 3600:
            print(f"[WARN] Skipping long window in row {index}: > 8 hours")
            continue

        st = client.get_waveforms(network='*', station='*', location='*', channel='*', starttime=stime, endtime=etime)

        # Filter traces by network
        st = st.select(network=tuple(valid_networks))

        if len(st) > 0:
            ids = sorted(set(tr.id for tr in st))
            df.at[index, 'SEED_ids'] = ','.join(ids)
            print(f"[INFO] Row {index}: found {len(ids)} valid SEED ids")

    except Exception as e:
        print(f"[ERROR] Row {index}: {e}")
        continue

# Save updated CSV
df.to_csv(output_path, index=False)
print(f"[DONE] Saved updated CSV to: {output_path}")