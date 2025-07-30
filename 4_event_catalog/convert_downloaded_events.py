import os
from obspy import read, Stream
import numpy as np
from datetime import datetime

'''
I think this is obsolete. it was used one time to chomp through the KSC EVENTS folders, and convert the SEED and MiniSEED files
to SDS filename convention into a folder called /data/KSC/eventminiseedfiles. but then i ran a subsequent function in fix_SDS_archive to
check these names again, and merge with the SDS archive
'''

# Function to check if the sampling rate matches the channel naming convention
def validate_channel_and_sampling_rate(channel, sampling_rate):
    # SEED channel convention and corresponding expected sampling rates
    # This is a basic example, and you can extend it with more cases
    channel_mapping = {
        'B': 75.0,
        'E': 100.0,
        'H': 100.0,
        'D': 250.0,
        'G': 2000.0 
    }

    bandcode = channel[0]
    if bandcode in 'SECF':
        if sampling_rate >= 20.0 and sampling_rate < 80.0:
            channel = 'S' + channel[1:]
        if sampling_rate >= 80.0 and sampling_rate < 250.0:
            channel = 'E' + channel[1:]  
        if sampling_rate >= 250.0 and sampling_rate < 1000.0:
            channel = 'C' + channel[1:]  
        if sampling_rate >= 1000.0 and sampling_rate < 5000.0:
            channel = 'F' + channel[1:] 
    if bandcode in 'BHDG':
        if sampling_rate >= 20.0 and sampling_rate < 80.0:
            channel = 'B' + channel[1:]
        if sampling_rate >= 80.0 and sampling_rate < 250.0:
            channel = 'H' + channel[1:]  
        if sampling_rate >= 250.0 and sampling_rate < 1000.0:
            channel = 'D' + channel[1:]  
        if sampling_rate >= 1000.0 and sampling_rate < 5000.0:
            channel = 'G' + channel[1:]                                      

    return channel

# Function to process each SEED/MiniSEED file
def process_miniseed_files(root_dir, outdir):
    # Traverse directories
    for root, dirs, files in os.walk(root_dir):
        streams_by_network_station = {}
        for file in files:
            if file.endswith(".mseed") or file.endswith(".seed"):  # Look for SEED/MiniSEED files
                miniseed_file = os.path.join(root, file)
                print(f"Processing file: {miniseed_file}")
                
                # Load the file into an obspy Stream object
                try:
                    st = read(miniseed_file)
                except Exception as e:
                    print(f"Error reading file {miniseed_file}: {e}")
                    continue
                
                # Process each Trace in the Stream
                for tr in st:
                    if len(tr.data)==0 or np.all(tr.data==0):
                        continue
                    network = tr.stats.network
                    #if not network in ['FL', 'AM', '1R']:
                    #    continue
                    station = tr.stats.station
                    
                    if len(tr.stats.location)==1:
                        tr.stats.location='0'+tr.stats.location
                    location = tr.stats.location or ""  # Default to empty string if no location
                    channel = tr.stats.channel
                    sampling_rate = tr.stats.sampling_rate
                    start_time = tr.stats.starttime
                    
                    # Validate the channel and sampling rate
                    tr.stats.channel = validate_channel_and_sampling_rate(channel, sampling_rate)
                    '''
                    # Create the output filename according to the convention
                    year = start_time.year
                    julian_day = start_time.julday
                    output_filename = f"{network}.{station}.{location}.{channel}.D.{year}.{int(julian_day):03d}.mseed"
                    output_file = os.path.join(outdir, output_filename)
                    ext = '.mseed'
                    count = 0
                    while os.path.isfile(output_file):
                        count += 1
                        newext = f'_{count}.mseed'
                        output_file = os.path.join(outdir, output_filename.replace(ext, newext))
                    
                    # Write the trace to a new MiniSEED file
                    tr.write(output_file, format="MSEED")
                    print(f"Written: {output_file}")
                    '''

                    # Create a unique key for each network.station.location.channel combination
                    trace_key = (network, station, location, channel)
                    
                    # Add the trace to the stream dictionary
                    if trace_key not in streams_by_network_station:
                        streams_by_network_station[trace_key] = Stream()
                    streams_by_network_station[trace_key].append(tr)
    
        # After processing all files, now merge traces for each group and write to file
        for (network, station, location, channel), stream in streams_by_network_station.items():

            # Get the start time from the first trace after merge (it should be the same)
            merged_trace = stream[0]
            year = merged_trace.stats.starttime.year
            julian_day = merged_trace.stats.starttime.julday
            
            # Create the output filename according to the convention
            output_filename = f"{network}.{station}.{location}.{channel}.D.{year}.{int(julian_day):03d}"
            output_file = os.path.join(outdir, output_filename)

            if os.path.isfile(output_file):
                continue


            # Merge the traces for the same network.station.location.channel on the same day
            try:
                stream.merge(fill_value=0)
            except:
                newstream = Stream()
                for count, tr in enumerate(stream):
                    newstream.append(tr)
                    try:
                        newstream.merge(fill_value=0)
                    except:
                        output_filename1 = f"{network}.{station}.{location}.{channel}.D.{year}.{int(julian_day):03d}_{count}"
                        output_file1 = os.path.join(outdir, output_filename1)
                        tr.write(output_file1, format='MSEED')
                        newstream.remove(tr)


            

            
            # Write the merged trace to a new MiniSEED file
            stream.write(output_file, format="MSEED")
            print(f"Written: {output_file}")                    

# Define the root directory to start searching for SEED/MiniSEED files
root_directory = "/data/KSC/events"
outdir = "/data/KSC/eventminiseedfiles"
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Process the files
process_miniseed_files(root_directory, outdir)

