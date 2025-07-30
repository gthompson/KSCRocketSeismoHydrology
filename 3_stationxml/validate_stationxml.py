from obspy import read_inventory
from obspy.core.inventory.util import Site
import sys

def validate_stationxml(path):
    try:
        inv = read_inventory(path)
        print(f"[SUCCESS] Parsed StationXML: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to parse StationXML:\n{e}")
        sys.exit(1)

    issues = []

    for net in inv:
        for sta in net:
            if not sta.site or not sta.site.name:
                issues.append(f"Missing site name for station {net.code}.{sta.code}")
            if not sta.site.description:
                issues.append(f"Missing site description for station {net.code}.{sta.code}")
            if not sta.channels:
                issues.append(f"No channels found for station {net.code}.{sta.code}")
            for cha in sta:
                if not cha.response:
                    issues.append(f"No response info for {net.code}.{sta.code}.{cha.location_code}.{cha.code}")
                if not cha.code or not cha.sample_rate:
                    issues.append(f"Incomplete channel info for {net.code}.{sta.code}.{cha.location_code}.{cha.code}")

    if issues:
        print("[WARNING] The following issues were found in the StationXML:")
        for issue in issues:
            print(" -", issue)
    else:
        print("[OK] No structural issues detected.")

# Example usage:
if __name__ == "__main__":
    stationxml_file = sys.argv[1] if len(sys.argv) > 1 else "station.xml"
    validate_stationxml(stationxml_file)
