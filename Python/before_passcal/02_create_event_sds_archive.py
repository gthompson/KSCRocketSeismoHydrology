import os
import glob
import pandas as pd
import numpy as np
from obspy import read, UTCDateTime, Stream
from flovopy.core.sds.SDS import SDSobj
from flovopy.core.preprocessing import fix_trace_id


def expand_channels(df):
    expanded_rows = []

    for idx, row in df.iterrows():
        chan = row["channel"]
        if isinstance(chan, str) and len(chan) > 3:
            basechan = chan[0:2]
            for channum in range(len(chan)-2):
                newchan = basechan + chan[channum+2]
                new_row = row.copy()
                new_row["channel"] = newchan
                expanded_rows.append(new_row)

    expanded_df = pd.DataFrame(expanded_rows)
    df = df[df["channel"].apply(lambda x: len(str(x)) == 3)]
    df = pd.concat([df, expanded_df], ignore_index=True)

    return df

# --- Step 1: Load the station metadata from Excel ---
excel_path = "/home/thompsong/Dropbox/DATA/station_metadata/ksc_stations_master_v2.xls"
df = pd.read_excel(excel_path, dtype={"location": str})

# --- Step 2: Prepare the DataFrame for lookup ---
df["ondate"] = pd.to_datetime(df["ondate"], errors='coerce')
df["offdate"] = pd.to_datetime(df["offdate"], errors='coerce')
df["ondate"] = df["ondate"].apply(lambda x: UTCDateTime(x) if pd.notnull(x) else None)
df["offdate"] = df["offdate"].apply(lambda x: UTCDateTime(x) if pd.notnull(x) else None)
df = expand_channels(df)

TOP_INPUT_DIR = '/data/KSC/event_files_to_convert_to_sds/'
sds = SDSobj("/data/SDS_EVENTS")
alreadyexists = []
unmatched_ids = {}

matches = 0
unmatches = 0

for f in sorted(glob.glob(os.path.join(TOP_INPUT_DIR, '*'))):
    st = Stream()
    this_st = read(f)
    for tr in this_st:
        if tr.stats.sampling_rate < 50.0 or tr.stats.station == 'LLS02':
            continue
        fix_trace_id(tr)
        if tr.stats.station == 'CARL0':
            tr.stats.station = 'BCHH'
        if tr.stats.station == '378':
            tr.stats.station = 'DVEL1'
        if tr.stats.station == 'FIRE' and tr.stats.starttime.year == 2018:
            tr.stats.station = 'DVEL2'

        if tr.stats.network == 'FL':
            tr.stats.network = '1R'

        if tr.stats.location in ['00', '0', '--', '', '10']:
            tr.stats.location = '00'

        st.append(tr)

    try:
        st.merge(fill_value=0, method=0)
    except:
        pass

    for tr in st:
        net = tr.stats.network
        sta = tr.stats.station
        cha = tr.stats.channel
        start = tr.stats.starttime
        end = tr.stats.endtime

        match = df[
            (df["network"] == net) &
            (df["station"] == sta) &
            (df["channel"] == cha) &
            (df["ondate"] <= start) &
            (df["offdate"] >= end - 86400)
        ]

        if not match.empty:
            location = match.iloc[0]["location"]
            tr.stats.location = str(location).zfill(2)
            matches += 1
        else:
            ymd = start.strftime('%Y-%m-%d')
            if tr.id in unmatched_ids:
                if ymd not in unmatched_ids[tr.id]:
                    unmatched_ids[tr.id].append(ymd)
                    unmatches += 1
            else:
                unmatched_ids[tr.id] = [ymd]
                unmatches += 1
            continue

    for tr in st:
        sds.write(tr)

    del st

print(f'Matches: {matches}, Unmatches {unmatches}')
print('\nUnmatched IDs:\n')
for key in unmatched_ids:
    print(key, '->', unmatched_ids[key])
