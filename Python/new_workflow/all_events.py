import pandas as pd
from obspy import UTCDateTime, Stream
from obspy.clients.filesystem.sds import Client
import numpy as np
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
minimum_sample_rate = 50.0

for index, row in df.iterrows():
    try:
        stime = UTCDateTime(row['window_start'])-60
        etime = UTCDateTime(row['window_end'])
        if etime <= stime:
            etime = stime + 60 * 15

        if (etime - stime) > 8 * 3600:
            print(f"[WARN] Skipping long window in row {index}: > 8 hours")
            continue

        st = client.get_waveforms(network='*', station='*', location='*', channel='*', starttime=stime, endtime=etime)

        # Filter traces by network
        st = Stream(tr for tr in st if tr.stats.network in valid_networks)
        for tr in st:
            if tr.stats.sampling_rate < minimum_sample_rate:
                st.remove(tr)

        if len(st) > 0:
            ids = sorted(set(tr.id for tr in st))
            df.at[index, 'SEED_ids'] = ','.join(ids)
            print(f"[INFO] Row {index}: found {len(ids)} valid SEED ids in /data/SDS_SAFE")
            eventdf = pd.DataFrame()
            eventdf['id']=[tr.id for tr in st]
            eventdf['min']=[np.nanmin(tr.data) for tr in st]
            eventdf['max']=[np.nanmax(tr.data) for tr in st]
            eventdf.to_csv(f'/data/KSC/event_seed_ids/launch_{stime}.csv',index=False)
            st.detrend('linear')
            st.filter('highpass', freq=0.5)
            st.plot(equal_scale=False, outfile=f'/data/KSC/event_seed_ids/launch_{stime}.png')
        else:
            print(f"[INFO] Row {index}: found 0 valid SEED ids in /data/SDS_SAFE")

    except Exception as e:
        print(f"[ERROR] Row {index}: {e}")
        continue

# Save updated CSV
df.to_csv(output_path, index=False)
print(f"[DONE] Saved updated CSV to: {output_path}")