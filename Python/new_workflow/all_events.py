from obspy import read, UTCDateTime

from obspy.clients.filesystem.sds import Client
client = Client('/data/SDS_SAFE')
'''
st = client.get_waveforms('FL', '*', '*', '*', UTCDateTime(2016,2,5,13,35,0), UTCDateTime(2016,2,5,13,45,0))
print(st)
st.detrend('linear')
st.filter('bandpass', freqmin=1.0, freqmax=10.0, corners=4)
st.plot(equal_scale=False)
'''

import pandas as pd

df = pd.read_csv('/home/thompsong/Developer/KSCRocketSeismoHydrology/Python/new_workflow/all_florida_launches.csv')
#df['ids']=''
for index, row in df.iterrows():
    stime=UTCDateTime(row['window_start'][0:19])
    etime=UTCDateTime(row['window_end'][0:19])
    print(stime, etime)
    if abs(etime-stime)>60*60*8:
        print('times are weird')
        continue    
    try:
        st = client.get_waveforms('*', '*', '*', '*', stime, etime)
        if len(st)>0:
            print(st)
            ids = []
            for tr in st:
                ids.append(tr.id)
            ids=sorted(list(set(ids)))
            row['ids']=ids
        print(row)
    except:
        print(f'Failed on row {index}')

