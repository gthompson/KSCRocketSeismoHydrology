# deploy website
import os
import shutil
import pandas as pd
thisdir = os.path.dirname(os.path.abspath(__file__))
srcdir = os.path.join(thisdir, 'website')
destdir= '/var/www/html/usfseismiclab.org/html/rocketcat'
if os.path.isdir(destdir):
    shutil.rmtree(destdir)
#df = pd.read_csv('/shares/hal9000/share/data/KSC/EROSION/DropboxFolder/2022_DATA/PilotStudy_KSC_Rocket_Launches.csv', index_col=None)
#df = df[['Date', 'Time', 'SLC', 'Rocket_Payload']]
#df.dropna(how='any', inplace=True)
#df.to_csv(os.path.join(srcdir, 'launches.csv'), index=False)    
shutil.copytree(srcdir, destdir)
#shutil.copy(os.path.join(thisdir, 'transducer_metadata_new.csv'), os.path.join(destdir, 'transducer_metadata.csv'))