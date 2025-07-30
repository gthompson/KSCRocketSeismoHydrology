from flovopy.sds.write_sds_archive_multiprocessing import write_sds_archive
import os

def list_all_files(root_dir):
    file_paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            file_paths.append(full_path)
    return file_paths



def main():

    #src_dir='/raid/data/KennedySpaceCenter/beforePASSCAL/CONTINUOUS/2016'
    src_dir='/raid/newhome/thompsong/work/PROJECTS/MASTERING/seed/MV'
    dest_sds='/raid/data/SDS_Montserrat'
    
    #excel_metadata_file = "/home/thompsong/Developer/KSCRocketSeismoHydrology/station_metadata/ksc_stations_master_v2.xlsx"
    # Example usage
    all_files = list_all_files(src_dir)
    print(f"Found {len(all_files)} files")

    # Optional filtering parameters (customize if needed)
    networks =  ['MV']
    stations = '*'
    start_date = "1995-01-01"      # e.g., "2020-01-01"
    end_date = "2026-01-10"        # e.g., "2022-01-01"

    # Now process the 2016-8 data which is already in an SDS archive
    write_sds_archive(
        src_dir=src_dir,
        dest_dir=dest_sds,
        networks=networks,
        stations=stations,
        start_date=start_date,
        end_date=end_date,
        metadata_excel_path=None,
        use_sds_structure=False,
        custom_file_list=all_files,
        #recursive=True,
        #file_glob="MV.*",
        n_processes=10,
        debug=False,                 
    )

if __name__ == "__main__":
    main()
