# List TOB3 files
import os
import glob
import pandas as pd

def process_tob3file(tob3file):
    print(tob3file)
    startdatetime=None
    with open(tob3file, 'rb') as f:
        firstline = f.readline().decode()
    #print(firstline)
    fields = firstline.split(',')
    for field in fields:
        if len(field)>5 and field[0:5]=='\"2022':
            startdatetime = field[1:-2].replace('\"', '')
    thisdict = {'file':os.path.basename(tob3file), 'datetime':startdatetime}
    return thisdict


lod = []
tob3dir = '/home/thompsong/Dropbox/PROFESSIONAL/RESEARCH/3_Project_Documents/NASAprojects/201602_Rocket_Seismology/DATA/2022_DATA/WellData/Uploads'
for uploaddir in sorted(glob.glob(os.path.join(tob3dir, '202*'))):
    for dirorfile in sorted(glob.glob(os.path.join(uploaddir, '*'))):
        if os.path.isdir(dirorfile):
            sampratedir = dirorfile
            for tob3file in sorted(glob.glob(os.path.join(sampratedir, '*.dat'))):
                thisdict = process_tob3file(tob3file)
                thisdict['uploaddir']=os.path.basename(uploaddir)
                thisdict['sampratedir']=os.path.basename(sampratedir)
                lod.append(thisdict)

        elif dirorfile[-4:] == '.dat':
            sampratedir = ''
            tob3file = dirorfile
            thisdict = process_tob3file(tob3file)
            thisdict['uploaddir']=os.path.basename(uploaddir)
            thisdict['sampratedir']=os.path.basename(sampratedir)
            lod.append(thisdict)
  
df = pd.DataFrame(lod)
df.sort_values(by='datetime')
df.to_csv('list_of_tob3_files.csv', index=False)