#!/usr/bin/env python3
import os
import sys
from flovopy.sds.write_sds_archive2 import write_sds_archive, check_for_collisions
import pandas as pd
import glob
import obspy
from flovopy.sds.sds import SDSobj
from flovopy.core.merge import smart_merge, read_mseed_with_gap_masking  #, restore_gaps


def main():

    #src_dir='/raid/data/KennedySpaceCenter/beforePASSCAL/CONTINUOUS/2016'
    src_dir='/data/KSC/beforePASSCAL/CONTINUOUS'
    dest_sds='/data/SDS_KSC'
    excel_metadata_file = "/home/thompsong/Dropbox/DATA/station_metadata/ksc_stations_master_v2.xls"
    log_file = os.path.join(dest_sds,"raid_data_KSC_C.log")
    csv_log = os.path.join(dest_sds,"raid_data_KSC_C.csv")

    # Optional filtering parameters (customize if needed)
    networks =  ['AM', '1R', 'XA', 'FL']
    stations = '*'
    start_date = None      # e.g., "2020-01-01"
    end_date = None        # e.g., "2022-01-01"
    write = True         # Set to False for dry-run

    #file_list = []
    TOP_INPUT_DIR = os.path.join(src_dir, '2016')
    sdsout = SDSobj(src_dir)
    for MMDIR in sorted(glob.glob(os.path.join(TOP_INPUT_DIR, '[0-9][0-9]'))):
        for DDDIR in sorted(glob.glob(os.path.join(MMDIR, '[0-9][0-9]'))):
            st = obspy.Stream()
            for f in sorted(glob.glob(os.path.join(DDDIR, '*FL*'))):
                #this_st = obspy.read(f)
                this_st = read_mseed_with_gap_masking(f, split_on_mask=True)
                        
                for tr in this_st:
                    st.append(tr)
            try:
                merged, status = smart_merge(st)
                print(status)
            except:
                print('failed to merge')
                continue
            sdsout.stream = merged
            sdsout.write()

    #print(f'File list for 2016 has {len(file_list)} files')
    
    '''
    write_sds_archive(
        src_dir=src_dir,
        dest_dir=dest_sds,
        networks=networks,
        stations=stations,
        start_date=start_date,
        end_date=end_date,
        write=write,
        log_file=log_file,
        metadata_excel_path=excel_metadata_file,
        csv_log_path=csv_log,
        use_sds_structure=False,
        custom_file_list=file_list          
    )

    check_for_collisions(csv_log)
    '''

    # Now process the 2016-8 data which is already in an SDS archive
    write_sds_archive(
        src_dir=src_dir,
        dest_dir=dest_sds,
        networks=networks,
        stations=stations,
        start_date=start_date,
        end_date=end_date,
        write=write,
        log_file=log_file,
        metadata_excel_path=excel_metadata_file,
        csv_log_path=csv_log,
        use_sds_structure=True,
        custom_file_list=None         
    )

    check_for_collisions(csv_log)


if __name__ == "__main__":
    main()