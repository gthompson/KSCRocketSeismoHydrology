#!/usr/bin/env python
import os
import obspy
import glob
import pandas as pd
import sys
sys.path.append('..')
import libWellData as LLE
print(os.getcwd())
os.chdir('../..')
#from create_data_inventory import uncalibrate_to_raw
paths={}
paths['transducersCSVfile']=os.path.join('transducers_metadata.csv')
#transducersDF = pd.read_csv('transducer_metadata.csv')
transducersDF = LLE.get_transducers_dataframe(paths)
os.chdir('/data/KSC/EROSION/obsolete/miniseed/good')
print(transducersDF)

def Stream2csv(st, csvfile):
    stime = min([tr.stats.starttime for tr in st])
    etime = max([tr.stats.endtime for tr in st])
    st.trim(starttime=stime, endtime=etime, pad=True, fill_value=0.0)
    dt = st[0].times('utcdatetime')
    df = pd.DataFrame()
    df['TIMESTAMP'] = dt
    for tr in st:
        serialnum = tr.stats.station + tr.stats.location
        df[serialnum] = tr.data
    print(df.head())

    LLE.uncalibrate_to_raw(transducersDF, df, csvfile)

    #df.to_csv(csvfile)
        

allfiles = sorted(glob.glob('*.ms'))
lod = []
for msfile in allfiles:
    parts = msfile.split('.')
    network = parts[0]
    station = parts[1] + parts[2]
    channel = parts[3]
    stime = obspy.UTCDateTime.strptime(parts[4], '%Y%m%d_%H%M%S').datetime
    etime = obspy.UTCDateTime.strptime(parts[5], '%Y%m%d_%H%M%S').datetime
    if channel[0]=='H':
        fsamp=100
    elif channel[0]=='B':
        fsamp=20
    elif channel[0]=='L':
        fsamp=1
 
    mydict = {'stime':stime, 'etime':etime, 'well':network, 'serialnum':station,'channel':channel,'msfile':msfile,'fsamp':fsamp}
    lod.append(mydict)
df = pd.DataFrame(lod)
#print(df)
df = df.sort_values(by=['stime'])
#print(df.head())
#dfg=df.groupby(['stime', 'fsamp'], as_index=False).size()
dfg=df.groupby(['stime', 'fsamp'])
for name,group in dfg:
    outfile=f'../Stream_{name[0].strftime("%Y%m%d%H%M%S")}_{name[1]}Hz.mseed'
    stall = obspy.Stream()
    print(name)
    #print(group)
    for i, row in group.iterrows():
        #print(row)
        st = obspy.read(row['msfile'])
        st.merge()
        for tr in st:
            stall.append(tr)
    stall.write(outfile,format='MSEED')
    Stream2csv(stall, outfile.replace('.mseed','.csv'))
