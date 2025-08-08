import os
import platform
from flovopy.stationmetadata.sensors import download_infraBSU_stationxml
from flovopy.stationmetadata.wrapper import get_stationXML_inventory
from flovopy.stationmetadata.convert import inventory2dataless_and_resp

# === Define metadata paths ===
home = os.path.expanduser("~")
system = platform.system()

# Use Dropbox path on macOS; otherwise default to /data
metadata_dir = os.path.join(home, 'Dropbox', 'DATA', 'station_metadata') if system == 'Darwin' else '/data/station_metadata'
os.makedirs(metadata_dir, exist_ok=True)

# Define all relevant paths
xml_file = os.path.join(metadata_dir, 'KSC2.xml')
resp_dir = os.path.join(metadata_dir, 'RESP2')
nrl_path = None #os.path.join(metadata_dir, 'NRL')
metadata_excel = os.path.join(metadata_dir, 'ksc_stations_master_v2.xlsx')
infraBSU_xml = os.path.join(metadata_dir, 'infraBSU_21s_0.5inch.xml')
stationxml_converter_jar = os.path.join(home, 'bin', 'stationxml-seed-converter.jar')

# Ensure output directories exist
os.makedirs(resp_dir, exist_ok=True)

# === Download infraBSU sensor XML if needed ===
if not os.path.isfile(infraBSU_xml):
    print(f"[INFO] Downloading infraBSU stationXML to {infraBSU_xml}")
    download_infraBSU_stationxml(save_path=infraBSU_xml)

# === Build full inventory with response ===
print("\n### Building full inventory with responses ###")

try:
    inventory = get_stationXML_inventory(
        xmlfile=xml_file,
        excel_file=metadata_excel,
        sheet_name='ksc_stations_master',
        infrabsu_xml=infraBSU_xml,
        nrl_path=nrl_path,
        overwrite=True,
        verbose=True
    )
except Exception as e:
    print(f"[ERROR] Failed to build inventory: {e}")
    inventory = None    


if inventory:

    print(f'\n\n********\nFinal inventory: {inventory}')

    # === Export to dataless SEED + RESP format ===
    print("\n### Exporting to dataless SEED and RESP ###")

    try:
        ############################################################################
        # Note: The stationxml-seed-converter.jar must be in the user's bin directory.
        # It can be downloaded from

        inventory2dataless_and_resp(
            inventory,
            output_dir=resp_dir,
            stationxml_seed_converter_jar=stationxml_converter_jar
        )
    except Exception as e:
        print(f"[ERROR] Failed to export to dataless SEED and RESP: {e}")
    else:
        print(f"[INFO] Successfully exported to {resp_dir}")
else:
    print("[ERROR] No inventory available to export.")