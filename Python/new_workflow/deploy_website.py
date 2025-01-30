# deploy website
import os
import shutil
import pandas as pd
thisdir = os.path.dirname(os.path.abspath(__file__))
srcdir = os.path.join(thisdir, 'website')
destdir= '/var/www/html/usfseismiclab.org/html/rocketcat'
if os.path.isdir(destdir):
    shutil.rmtree(destdir)
shutil.copytree(srcdir, destdir)


src_directory = '/shares/hal9000/share/data/KSC/EROSION/DropboxFolder/2022_DATA/EVENTS'

def copy_png_files_with_structure(src_dir, dest_dir):
    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if the file is a .png file
            if file.lower().endswith('.png'):
                # Compute the relative path of the file
                rel_path = os.path.relpath(root, src_dir)
                
                # Construct the destination directory by joining the destination base path with the relative path
                dest_dir_path = os.path.join(dest_dir, rel_path)
                
                # Ensure the destination directory exists
                if not os.path.exists(dest_dir_path):
                    os.makedirs(dest_dir_path)
                
                # Construct the source and destination file paths
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_dir_path, file)
                
                # Copy the file to the destination directory
                shutil.copy2(src_file, dest_file)
                print(f"Copied {src_file} to {dest_file}")

copy_png_files_with_structure(src_directory, os.path.join(destdir, 'EVENTS'))
csvsrc = os.path.join(src_directory, 'launchesDF.csv')
#csvsrc = '/shares/hal9000/share/data/KSC/EROSION/DropboxFolder/2022_DATA/PilotStudy_KSC_Rocket_Launches.csv'
csvtrg = os.path.join(destdir, 'launches.csv')
df = pd.read_csv(csvsrc, index_col=None)
df = df[['datetime', 'SLC', 'Rocket_Payload']]
df.dropna(how='all', inplace=True)
df.to_csv(csvtrg, index=False)    

