
import os
import sys
import re
repopath = os.path.join(os.getenv('HOME'), 'Developer', 'KSCRocketSeismoHydrology')
os.chdir(repopath)
sys.path.append('Python')
print(os.getcwd())
import glob
import pandas as pd
import obspy
from IPython.display import clear_output
RAWDIR = '/data/KSC/EROSION/fromdropboxinventory'
FILELOOKUPCSV = os.path.join(RAWDIR, 'reverse_lookup.csv')
filesnotprocessedfile = os.path.join(RAWDIR, 'files_not_processed.txt')
MASTERCSV = os.path.join(RAWDIR, f'data_inventory.csv')
erase=False
if erase:
    os.system(f'rm -rf {RAWDIR}/*')
    os.system(f'echo "fullpath, outpklfullpath, basename, starttime, endtime, seqno" > {FILELOOKUPCSV}')
lookupdf = pd.read_csv(FILELOOKUPCSV)
if os.path.isfile(MASTERCSV):
    allmasterrows = pd.read_csv(MASTERCSV).to_dict('records')
else:
    allmasterrows=[]
import libWellData as LLE
transducersDF = pd.read_csv('transducer_metadata.csv')

def uncalibrate_to_raw(df, pklfile, to_csv=True):
    print('- Reverse calibration equations')
    for col in df.columns:
        if col[0:2]=='12' or col[0:2]=='21':
            this_transducer = transducersDF[(transducersDF['serial']) == col]
            #print(this_transducer)
            if len(this_transducer.index)==1:
                this_transducer = this_transducer.iloc[0].to_dict()
                #print(this_transducer)
                df[col] = LLE.reverse_compute_psi(df[col].to_numpy(), this_transducer)
    print('- writing reverse corrected data to %s' % pklfile)
    if to_csv:       
        df.to_csv(pklfile.replace('.pkl', '.csv'), index=False)
    else:
        df.to_pickle(pklfile)

def copy_to_raw(df, pklfile, to_csv=True):
    print('- copying data to %s' % pklfile)       
    if to_csv:       
        df.to_csv(pklfile.replace('.pkl', '.csv'), index=False)
    else:
        df.to_pickle(pklfile)

def write_masterdf(allmasterrows):
    if len(allmasterrows)>0:
        masterdf = pd.DataFrame(allmasterrows)
        print(f'Writing/updating {MASTERCSV}')
        if 'starttime' in masterdf.columns:
            masterdf.sort_values(by=['starttime','uploadfolder','sampratefolder','samprate']).to_csv(MASTERCSV, index=False)
        else:
            masterdf.to_csv(MASTERCSV, index=False)

def process_file(dirpath, file, filenum, load=False):
    clear_output()
    print(f'Processing file {filenum}: {file} in {dirpath}')
    fparts = file.split('.')
    if len(fparts)==4:
        uploadfolder, sampratefolder, basefilename, ext = fparts
    elif len(fparts)==2:
        uploadfolder = 'unknown'
        sampratefolder = 'unknown'
        basefilename, ext = fparts
        dparts = dirpath.split('/')
        if len(dparts)>3:
            uploadfolder = dparts[-2]
            sampratefolder = dparts[-1]
    try:
        sampratefolder1, middlename, realsamprateandseqno = basefilename.split('_')
    except:
        print('Failed to split: ',basefilename)
        return 'failed to split basename'
    if sampratefolder.lower() == sampratefolder1.lower(): # not a Baro file
        if 'Hz' in realsamprateandseqno:
            realsamprate, seqno = realsamprateandseqno.split('Hz')
        elif 'Sec' in realsamprateandseqno:
            realsamprate, seqno = realsamprateandseqno.split('Sec')
        else:
            print(f'Did not find Hz or Sec in filename: {file}')
            return 'Did not find Hz or Sec in filename'
    elif sampratefolder == 'Baro':
        realsamprate, baro, seqno = basefilename.split('_')
        realsamprate = realsamprate.split('hz')[0]
        seqno = seqno.split('Sensors')[-1]
    else:
        print(f'samprate do not match: {file}, {sampratefolder}, {sampratefolder1}' +'\n')
        return 'samprate do not match'
    #print(realsamprate, seqno)
    #masterrow={'filename':os.path.basename(file), 'topdir':dirpath, 'uploadfolder':os.path.basename(uploadfolder), 'sampratefolder':sampratefolder, \
    #           'basename':basefilename, 'samprate':realsamprate, 'seqno':seqno} 
    masterrow={'filename':file, 'topdir':dirpath, 'uploadfolder':uploadfolder, 'sampratefolder':sampratefolder, \
               'basename':basefilename, 'samprate':realsamprate, 'seqno':seqno} 


    if load:
        dropped_headers = False
        dropped_rows = 0
        fullpath = os.path.join(dirpath, file)
        print(f'Loading {fullpath}')
        try:
            if file.endswith('pkl'):
                df = pd.read_pickle(fullpath)
            elif file.endswith('csv'):
                df = pd.read_csv(fullpath)
        except Exception as e:
            print(e)
            os.system(f'head {fullpath}')
            raise e
        
        ''' Note that converted TOB3 files have multiple header lines and columns are read as dtype "object" because of mixed dtype. So we have to explicity convert them after removing excess header rows.
            The first header row is garbage. We want the second '''


        # Drop incorrect header row
        columns_old = []
        if not 'TIMESTAMP' in df.columns: # use 2nd row of file/0th row of dataframe for columns instead
            columns_old = df.columns
            df.columns = df.iloc[0]
            df=df[1:]
            dropped_headers=True
            
        # Convert TIMESTAMP
        df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='ISO8601', errors='coerce')
        l1 = len(df)
        df = df.dropna(subset=['TIMESTAMP'])

        # filter by TIMESTAMP further
        TS_median = df['TIMESTAMP'].median()
        TS_start = TS_median - pd.Timedelta(hours=4)
        TS_end = TS_median + pd.Timedelta(hours=4)
        df = df[(df['TIMESTAMP'] > TS_start ) & (df['TIMESTAMP'] < TS_end)]
        l2 = len(df)
        dropped_rows=l1-l2
        masterrow['starttime'] = df['TIMESTAMP'].min()
        masterrow['endtime'] = df['TIMESTAMP'].max()
        masterrow['dropped_headers']=dropped_headers
        masterrow['dropped_rows']=dropped_rows

        # Drop empty columns
        df = df.dropna(axis=1, how='all')    
        masterrow['calibrated'] = None
        for colnum, col in enumerate(df.columns):
            if col!='TIMESTAMP':
                if 'Unnamed' in col: 
                    if dropped_headers: # if for TOB3 converted files, there is no column header on second row, we try the first row again DO NOT THINK THIS EVER HAPPENS
                        df.rename(columns=[col, columns_old[colnum]], inplace=True)
                        col = columns_old[colnum]
                    else:
                        continue
                        # just seems to be an index that was saved into CSV file in corrected directory.
                        #if col=='Unnamed: 0' and df.loc[0][col]=='3':
                        #    continue
                            
                try:
                    df[col]=df[col].astype(float)
                    if df[col].apply(float.is_integer).all():
                        df[col]=df[col].astype(int)
                        masterrow[col] = int(df[col].median())
                    else:
                        masterrow[col] = df[col].median()
                except Exception as e:
                    print(e)
                    print(f'Failed to convert column {col}')
                    print(df[col])
                    exit()
                if col[0:2]=='12' or col[0:2]=='21': # range seems to be 8000-10000 if not converted to calibrated units
                    if masterrow[col] > 7000.0:
                        masterrow['calibrated'] = False
                    else:
                        masterrow['calibrated'] = True
            #print(col, df[col].dtype)
        #mybasename = re.sub(f'masterrow["seqno"]$', '', masterrow['basename'])
        mybasename = masterrow['basename'][:masterrow['basename'].rfind(masterrow['seqno'])]
        outpkldir = os.path.join(RAWDIR, masterrow['uploadfolder'], masterrow['sampratefolder'])
        '''
        outpklfile = masterrow['topdir'].replace('./','').replace('/','_') + '_' + \
            mybasename + '_' + \
            masterrow['starttime'].strftime('%Y%m%d%H%M%S_') + \
            f"{int(masterrow['seqno']):03d}" + \
            '.pkl'
        '''
        outpklfile = mybasename + '_' + \
            masterrow['starttime'].strftime('%Y%m%d%H%M%S_') + \
            f"{int(masterrow['seqno']):03d}" + \
            '.pkl'
        outpklfullpath = os.path.join(outpkldir, outpklfile)
        if not os.path.isdir(outpkldir):
            os.makedirs(outpkldir)
        if masterrow['calibrated']==True:
            outpklfullpath = outpklfullpath.replace('.pkl', '_REVERSED.pkl')
        while os.path.isfile(outpklfullpath):
            outpklfullpath = outpklfullpath.replace('.pkl', 'x.pkl')
        if masterrow['calibrated']==True:
            uncalibrate_to_raw(df, outpklfullpath)
        else:
            copy_to_raw(df, outpklfullpath)
        os.system(f"echo {fullpath}, {outpklfullpath}, {masterrow['basename']}, {masterrow['starttime']}, {masterrow['endtime']}, {masterrow['seqno']} >> {FILELOOKUPCSV}")
        
    return masterrow
  
pwd = os.getcwd()
os.chdir(os.path.join(os.getenv('HOME'), 'Dropbox/PROFESSIONAL/RESEARCH/3_Project_Documents/NASAprojects/201602_Rocket_Seismology/DATA/2022_DATA/WellData'))
print(os.listdir())
if os.path.isfile(filesnotprocessedfile):
    os.unlink(filesnotprocessedfile)
masterdf=pd.DataFrame()

filenum = 0
for dirpath, dirnames, filenames in os.walk("."):
    if 'combined' in dirpath:
        continue
    for filename in sorted(filenames):
        if filename.endswith((".csv", '*.pkl')):
            masterrow = []
            if 'gps' in filename or 'data_inventory' in filename or 'lookuptable' in filename or 'transducer' in filename or 'HOF' in filename:
                masterrow = 'did not match file filter'
            else:
                if filename.startswith("._"):
                    filename2 = filename[2:]
                    if filename2 in filenames:
                        masterrow = 'starts with ._'
            subsetdf = lookupdf[lookupdf['fullpath']==os.path.join(dirpath, filename)]
            if len(subsetdf)>0:
                masterrow='Processed already'

            if not masterrow:
                filenum += 1
                print(filenum, dirpath, filename)
                masterrow = process_file(dirpath, filename, filenum, load=True)
            if isinstance(masterrow, dict):
                allmasterrows.append(masterrow)
                if filenum % 100 == 0:
                    write_masterdf(allmasterrows)
            else:
                os.system(f"echo {os.path.join(dirpath, filename)}: {masterrow} >> {filesnotprocessedfile}")
write_masterdf(allmasterrows)                
os.chdir(pwd)