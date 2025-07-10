#!/usr/bin/env python3

import sys
from flovopy.core.sds.fix_sds_archive import fix_sds_archive

def main():
    # Define input/output paths and metadata file
    src_sds = "/data/SDS"
    dest_sds = "/data/SDS_CONTINUOUS"
    excel_metadata_file = "/home/thompsong/Dropbox/DATA/station_metadata/ksc_stations_master_v2.xls"

    # Optional filtering parameters (customize if needed)
    networks =  ['AM', '1R', 'XA', 'FL']
    stations = '*'
    start_date = None      # e.g., "2020-01-01"
    end_date = None        # e.g., "2022-01-01"
    write = True           # Set to False for dry-run
    log_file = "fix_sds_archive.log"

    # Call the function
    fix_sds_archive(
        src_dir=src_sds,
        dest_dir=dest_sds,
        networks=networks,
        stations=stations,
        start_date=start_date,
        end_date=end_date,
        write=write,
        log_file=log_file,
        metadata_excel_path=excel_metadata_file
    )

if __name__ == "__main__":
    main()