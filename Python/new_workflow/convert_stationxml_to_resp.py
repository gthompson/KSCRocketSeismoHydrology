#!/usr/bin/env python

import os
import sys
from obspy import read_inventory
from obspy.io.xseed import resp

def convert_stationxml_to_resp(stationxml_path, output_dir):
    try:
        inv = read_inventory(stationxml_path)
    except Exception as e:
        print(f"[ERROR] Failed to read StationXML file: {e}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    count = 0

    for network in inv:
        for station in network:
            for channel in station:
                net = network.code
                sta = station.code
                loc = channel.location_code or "--"
                cha = channel.code

                filename = f"RESP.{net}.{sta}.{loc}.{cha}".replace("--", "")
                filepath = os.path.join(output_dir, filename)

                try:
                    blist = channel.response.get_response_blockettes()
                    with open(filepath, "w") as f:
                        f.write(resp.Response(blockette_list=blist).get_resp())
                    print(f"[OK] Wrote {filepath}")
                    count += 1
                except Exception as e:
                    print(f"[WARNING] Skipped {net}.{sta}.{loc}.{cha}: {e}")

    print(f"[INFO] Created {count} RESP files in '{output_dir}'")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:\n  python convert_stationxml_to_resp.py <StationXML> <Output_DIR>")
        sys.exit(1)

    stationxml_path = sys.argv[1]
    output_dir = sys.argv[2]

    convert_stationxml_to_resp(stationxml_path, output_dir)
