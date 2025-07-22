from flovopy.sds.write_sds_archive_multiprocessing import write_sds_archive
import os 
import glob

def main():

    #src_dir='/raid/data/KennedySpaceCenter/beforePASSCAL/CONTINUOUS/2016'
    src_dir='/data/KSC/beforePASSCAL/CONTINUOUS'
    dest_sds='/data/SDS_hourlyKSChal'
    
    excel_metadata_file = "/home/thompsong/Dropbox/DATA/station_metadata/ksc_stations_master_v2.xls"

    # Optional filtering parameters (customize if needed)
    networks =  ['AM', '1R', 'XA', 'FL']
    stations = '*'
    start_date = "2016-01-01"      # e.g., "2020-01-01"
    end_date = "2026-01-10"        # e.g., "2022-01-01"


    file_list = []
    for YYDIR in sorted(glob.glob(os.path.join(src_dir, '20[0-9][0-9]'))):
        for MMDIR in sorted(glob.glob(os.path.join(YYDIR, '[0-9][0-9]'))):
            for DDDIR in sorted(glob.glob(os.path.join(MMDIR, '[0-9][0-9]'))):
                for f in sorted(glob.glob(os.path.join(DDDIR, '*FL*'))):
                    file_list.append(f)

    write_sds_archive(
        src_dir=src_dir,
        dest_dir=dest_sds,
        networks=networks,
        stations=stations,
        start_date=start_date,
        end_date=end_date,
        metadata_excel_path=excel_metadata_file,
        use_sds_structure=False,
        custom_file_list=file_list,
        #recursive=True,
        #file_glob="*.mseed",
        n_processes=6,
        debug=False,                 
    )    


if __name__ == "__main__":
    main()