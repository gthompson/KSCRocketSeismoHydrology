import os
import glob
import pandas as pd
import numpy as np
from obspy import read, UTCDateTime, Stream
from flovopy.core.sds.SDS import SDSobj
from flovopy.core.preprocessing import fix_trace_id

TOP_INPUT_DIR = '/data/KSC/event_files_to_convert_to_sds/'
excel_metadata_file = "/home/thompsong/Dropbox/DATA/station_metadata/ksc_stations_master_v2.xls"
SDS_TOP = "/data/SDS_EVENTS"
sds = SDSobj(SDS_TOP)
sds.load_metadata_from_excel(excel_metadata_file)

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
        st.append(tr)

    try:
        st.merge(fill_value=None, method=0)
    except:
        pass

    for tr in st:
        match = sds.match_metadata(tr)

        if match is not None:
            location = match["location"]
            tr.stats.location = str(location).zfill(2)
            matches += 1
        else:
            ymd = tr.stats.starttime.strftime('%Y-%m-%d')
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
