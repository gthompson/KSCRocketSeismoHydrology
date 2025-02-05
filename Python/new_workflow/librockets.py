import numpy as np
import obspy
import os
import sys
import matplotlib.pyplot as plt
import header
paths = header.setup_environment()
sys.path.append(os.path.join(paths['Developer'], 'SoufriereHillsVolcano', 'lib'))
import libseisGT



def detectEvent(st):
    st2 = st.copy().filter('bandpass', freqmin=5, freqmax=16)
    return libseisGT.detect_network_event(st2, threshon=4, threshoff=1, sta=5, lta=100, pad=0.0, best_only=True)

def sds2eventStream(launchtime, sdsclient, thisSDSobj, pretrig=3600, posttrig=3600):  
      
    startt = obspy.UTCDateTime(launchtime) - pretrig
    endt = obspy.UTCDateTime(launchtime) + posttrig
    st = try_different_waveform_loading_methods(sdsclient, thisSDSobj, startt, endt)
    return st

def try_different_waveform_loading_methods(sdsclient, thisSDSobj, startt, endt):

    # ObsPy SDS archive reader
    st3 = sdsclient.get_waveforms("*", "*", "*", "[HDCES]*", startt, endt)

    # My SDS class that wraps ObsPy SDS reader
    thisSDSobj.read(startt, endt, speed=1)
    st4 = thisSDSobj.stream

    combine_streams(st3, st4)    
    
    return st3
    
def combine_streams(stB, stA):
    appended = False
    for trA in stA:
        found = False
        for trB in stB:
            if trA.stats.station == trB.stats.station and trA.stats.location == trB.stats.location and trA.stats.channel == trB.stats.channel:
                if trA.stats.network == '':
                    trA.stats.network = trB.stats.network
                if trA.stats.starttime >= trB.stats.starttime and trA.stats.endtime <= trB.stats.endtime:
                    found = True
                    break
        if not found:
            stB.append(trA)
            appended = True
    if appended:
        stB.merge(method=0, fill_value=0)

'''
def clean(st, taperseconds):
    for tr in st:
        trace_seconds = tr.stats.delta * tr.stats.npts
        libseisGT.clean_trace(tr, taperFraction=taperseconds/trace_seconds, filterType="highpass", freq=0.01, corners=2, zerophase=True, inv=None)
    


def apply_calibration_correction(st):
    # calibration correction

    for tr in st:
        if 'countsPerUnit' in tr.stats:
            continue
        else:
            tr.stats['countsPerUnit'] = 1
            if not 'units' in tr.stats:
                tr.stats['units'] = 'Counts'
            if tr.stats.station[0].isnumeric(): # well data
                if len(tr.stats.network)==0:
                    tr.stats.network = '6'
                if tr.stats.channel[2] == 'D':
                    tr.stats.countsPerUnit = 1/LLE.psi2inches(1) # counts (psi) per inch
                    tr.stats.units = 'inches'
                elif tr.stats.channel[2] == 'H':
                    tr.stats.countsPerUnit = 1/6894.76 # counts (psi) per Pa
                    tr.stats.units = 'Pa'
            elif tr.stats.channel[1]=='D':
                tr.stats.countsPerUnit = 720 # counts/Pa on 1 V FS setting
                if tr.id[:-1] == 'FL.BCHH3.10.HD':
                    if tr.stats.starttime < obspy.UTCDateTime(2022,5,26): # Chaparral M25. I had it set to 1 V FS. Should have used 40 V FS. 
                        if tr.id == 'FL.BCHH3.10.HDF':
                            tr.stats.countsPerUnit = 8e5 # counts/Pa
                        else:
                            tr.stats.countsPerUnit = 720 # counts/Pa 
                    else: # Chaparral switched to 40 V FS
                        if tr.id == 'FL.BCHH3.10.HDF':
                            tr.stats.countsPerUnit = 2e4 # counts/Pa
                        else:
                            tr.stats.countsPerUnit = 18 # counts/Pa 
                tr.stats.units = 'Pa'

            elif tr.stats.channel[1]=='H':
                tr.stats.countsPerUnit = 3e2 # counts/(um/s)
                tr.stats.units = 'um/s'
            tr.data = tr.data/tr.stats.countsPerUnit


def maxamp(tr):
    return np.max(np.abs(tr.data))

def add_snr(st, assoctime, threshold=1.5):
    nstime = max([st[0].stats.starttime, assoctime-240])
    netime = min([st[0].stats.endtime, assoctime-60])
    sstime = assoctime
    setime = min([st[0].stats.endtime, assoctime+120])    
    for tr in st:
        tr_noise = tr.copy().trim(starttime=nstime, endtime=netime)
        tr_signal = tr.copy().trim(starttime=sstime, endtime=setime)
        tr.stats['noise'] = np.nanmedian(np.abs(tr_noise.data))
        tr.stats['signal'] = np.nanmedian(np.abs(tr_signal.data))
        tr.stats['snr'] = tr.stats['signal']/tr.stats['noise']

def group_streams_for_plotting(st):
    groups = {}
    stationsWELL = ['6S', '6I']
    for station in stationsWELL:
        stationStream = st.select(network=station)
        #stationIDS = list(set([tr.id for tr in stationStream]))
        groups[station] = stationStream
    streamSA = st.select(network='FL')
    stationsSA = list(set([tr.stats.station for tr in streamSA]))
    for station in stationsSA:
        stationStream = streamSA.select(station=station)
        #stationIDS = list(set([tr.id for tr in stationStream]))
        groups[station] = stationStream
    #print(groups)
    return groups 

def despike_trace(trace):
    # Parameters for despiking
    window_size = 5  # Size of the sliding window
    threshold_factor = 3  # Threshold factor to define spikes

    # Loop over the data and apply despiking
    despiked_data = trace.data.copy()
    for i in range(window_size, len(trace.data) - window_size):
        # Define the window of data around each point
        window = trace.data[i - window_size:i + window_size + 1]
        
        # Calculate local median and standard deviation
        local_median = np.median(window)
        local_std = np.std(window)
        
        # Identify spikes: data point deviates too much from the local median
        if np.abs(trace.data[i] - local_median) > threshold_factor * local_std:
            despiked_data[i] = local_median  # Replace with the local median

    # Update the trace data with the despiked data
    trace.data = despiked_data
'''

from obspy.signal.invsim import cosine_taper
from scipy.signal import hilbert, convolve
def get_envelope(tr, seconds=1):
    envelope = np.abs(hilbert(tr.data))
    # Smooth the envelope using a moving average filter
    window_size = int(tr.stats.sampling_rate*seconds)  # Adjust the window size based on your needs
    window = np.ones(window_size) / window_size  # Simple moving average kernel
    envelope = convolve(envelope, window, mode='same')   
    return envelope 

def plot_envelope2(st, window_size=1.0, percentile=99, outfile=None, units=None):
    # Plot the Nth percentile for each 1-second window
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green', 'orange', 'black', 'grey', 'purple', 'cyan']
    for i, tr in enumerate(st):

        # Calculate the number of samples per window (1 second)
        samples_per_window = int(window_size * tr.stats.sampling_rate)

        # Get the data and time values
        data = abs(tr.data)
        times = tr.times()

        # Reshape the data into windows (each row is a 1-second window)
        num_windows = len(data) // samples_per_window  # Number of full windows
        reshaped_data = data[:num_windows * samples_per_window].reshape((num_windows, samples_per_window))

        # Calculate the 99th percentile for each window along axis 1 (columns)
        percentiles = np.nanpercentile(reshaped_data, percentile, axis=1)

        # Create the corresponding time values for each window (mid-point of each window)
        window_times = times[:num_windows * samples_per_window:samples_per_window] + window_size / 2


        plt.plot(window_times, percentiles, label=tr.id, color=colors[i % len(colors)], lw=1)
    plt.title(f"{percentile}th Percentile in {window_size}-Second windows")
    plt.xlabel(f"Time (s) from {st[0].stats.starttime}")
    if units:
        plt.ylabel(units)
    plt.grid(True)
    plt.legend()
    if outfile:
        plt.savefig(fname=outfile)
    else:
        plt.show()

def plot_seismograms(st, outfile=None, bottomlabel=None, ylabels=None, units=None, channels='ZNE'):
    """ Create a plot of a Stream object similar to Seisan's mulplt """
    fh = plt.figure(figsize=(8,12))

    from cycler import cycler

    # Define a list of colors you want to cycle through
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Set the color cycle in matplotlib
    plt.rcParams['axes.prop_cycle'] = cycler(color=colors)    
    
    # get number of stations
    stations = []
    for tr in st:
        stations.append(tr.stats.station)
    stations = list(set(stations))
    n = len(stations)
    
    # start time as a Unix epoch
    startepoch = st[0].stats.starttime.timestamp
    
    # create empty set of subplot handles - without this any change to one affects all
    axh = []
    
    # loop over all stream objects
    #colors = ['black', 'blue', 'green']
    
    linewidths = [0.25, 0.1, 0.1, 0.25]
    for i in range(n):
        # add new axes handle for new subplot
        #axh.append(plt.subplot(n, 1, i+1, sharex=ax))
        if i>0:
            #axh.append(plt.subplot(n, 1, i+1, sharex=axh[0]))
            axh.append(fh.add_subplot(n, 1, i+1, sharex=axh[0]))
        else:
            axh.append(fh.add_subplot(n, 1, i+1))
        
        # find all the traces for this station
        this_station = stations[i]
        these_traces = st.copy().select(station=this_station)
        all_ys = []
        lw = 0.5
        if len(these_traces)==1:
            lw = 2
        for this_trace in these_traces:
            this_component = this_trace.stats.channel[2]
            line_index = channels.find(this_component)
            if line_index>-1:
                #print(this_trace.id, line_index, colors[line_index], linewidths[line_index])
            
                # time vector, t, in seconds since start of record section
                t = np.linspace(this_trace.stats.starttime.timestamp - startepoch,
                    this_trace.stats.endtime.timestamp - startepoch,
                    this_trace.stats.npts)
                #y = this_trace.data - offset
                y = get_envelope(this_trace, seconds=0.1)
                all_ys.append(y)
                
                # PLOT THE DATA
                axh[i].plot(t, y, lw=lw, color=colors[line_index], label=this_component)
                axh[i].autoscale(enable=True, axis='x', tight=True)
        if len(these_traces)==3:
                vector_amplitude = np.sqrt(all_ys[0]**2 + all_ys[1]**2 + all_ys[2]**2)
                axh[i].plot(t, vector_amplitude, color='red', label='vector', lw=2)
                axh[i].autoscale(enable=True, axis='x', tight=True)  
        elif len(these_traces)>1:
                # Stack the data from each trace
                data = np.array([y for y in all_ys])

                # Compute the median across traces (axis 0 means median across the first dimension - i.e., along traces)
                median_data = np.nanmedian(data, axis=0)
                axh[i].plot(t, median_data, color='red', label='median', lw=2)
                axh[i].autoscale(enable=True, axis='x', tight=True)                         
        plt.grid()
        plt.legend()
        

   
        # remove yticks because we will add text showing max and offset values
        #axh[i].yaxis.set_ticks([])
        '''
        # remove xticklabels for all but the bottom subplot
        if i < n-1:
            axh[i].xaxis.set_ticklabels([])
        else:
            # for the bottom subplot, also add an xlabel with start time
            if bottomlabel:
                plt.xlabel(bottomlabel)
            else:
                plt.xlabel("Starting at %s" % (st[0].stats.starttime) )
        '''

        # default ylabel is station.channel
        ylabelstr = this_station + '\n' + units
        if ylabels:
            ylabelstr = ylabels[i]
        plt.ylabel(ylabelstr, rotation=90)

        plt.xlabel(f'Seconds from {st[0].stats.starttime}')

    # change all font sizes
    plt.rcParams.update({'font.size': 10})
    
    plt.subplots_adjust(wspace=0.1)

    #plt.suptitle(f'Amplitude from {st[0].stats.starttime}')
    
    # show the figure
    if outfile:
        plt.savefig(outfile, bbox_inches='tight')    
    else:
        plt.show()
    

def floor_minute(timestamp):
    return timestamp.replace(second=0, microsecond=0) 

def ceil_minute(timestamp):
    return (timestamp+60).replace(second=0, microsecond=0) 

def super_stream_plot(st, dpi=100, Nmax=6, rank='ZFODNE123456789', equal_scale=True, outfile=None, figsize=(8,8)):
    fig = plt.figure(figsize=figsize)  # Set a custom figure size (width, height)
    if len(st)>Nmax:
        st2 = obspy.Stream()
        rankpos = 0
        while len(st2)<Nmax and rankpos<len(rank):
            for tr in st.select(channel=f'*{rank[rankpos]}'):
                if len(st2)<Nmax:
                    st2.append(tr)
            rankpos += 1
    else:
        st2 = st.copy()

    height = 250
    if len(st2)>4:
        height=1000/len(st2)
        #height=100
    #size_tuple = (1200, height) * len(st2)
    #size_tuple = (height, 1200) * len(st2)
    size_tuple = (1000, height) * len(st2)
    st2.plot(fig=fig, dpi=dpi, color='b', equal_scale=equal_scale, outfile=outfile);

def remove_response(seismic_st, output='VEL'):
    this_st = seismic_st.copy()
    try:
        this_st.remove_response(inventory=inv, output=output, pre_filt=None) #correct to velocity
    except:
        for tr in this_st:
            print(tr)
            try:
                tr.remove_response(inventory=inv, output=output, pre_filt=None) #correct to displacement
            except Exception as e:
                print(e)
                this_st.remove(tr)    
    return this_st


from sklearn.mixture import GaussianMixture
def detect_incorrect_p2p_voltage_setting(st, stationratio=20.0, allratio=200.0):
    # for a Centaur datalogger, this is designed to detect if p2p voltage should have been 1 rather than 40
    # we do this by checking for a bimodal distribution of the 95th percentile of the data in each trace

    amplitudes = np.array([np.nanpercentile(abs(tr.data), 95) for tr in st])
    m = np.nanmedian(amplitudes)
    for i, tr in enumerate(st):
        if amplitudes[i] > m * allratio or amplitudes[i] < m/allratio: # bad channel
            st.remove(tr) 


    stations = ['BCHH*', 'BHP*']

    for station in stations:
        print(station)
        st_station = st.select(station=station)
        if len(st_station)>2:
            amplitudes = np.array([np.nanpercentile(abs(tr.data), 95) for tr in st_station])
            print(amplitudes)

            # Fit a GMM to the data with correction factor 1
            gmm = GaussianMixture(n_components=2)
            gmm.fit(amplitudes.reshape(-1, 1))

            # Get the responsibilities (probabilities) of each data point for both components
            responsibilities = gmm.predict_proba(amplitudes.reshape(-1, 1))

            # Assign each data point to the component with the highest responsibility
            component_1_indices = np.where(responsibilities[:, 0] > responsibilities[:, 1])[0]
            component_2_indices = np.where(responsibilities[:, 1] > responsibilities[:, 0])[0]
            print(component_1_indices, component_2_indices)
            #print(st)

            '''
            # Plot the seismogram and color-code by component
            plt.figure(figsize=(10, 6))
            plt.plot(amplitudes, label="Corrected Amplitude", color='gray', alpha=0.6)

            # Plot the data points for each component with different colors
            plt.scatter(component_1_indices, amplitudes[component_1_indices], color='blue', label="Component 1", s=10)
            plt.scatter(component_2_indices, amplitudes[component_2_indices], color='red', label="Component 2", s=10)

            plt.title("Amplitude Distribution with GMM Components")
            plt.xlabel("Sample Index")

            plt.show()
            '''

            if len(component_2_indices)>0:
                print(gmm.means_)
                mratio = gmm.means_[1]/gmm.means_[0]
                if gmm.means_[0] > gmm.means_[1]:
                    for tr in [st[i] for i in component_1_indices]:
                        if mratio<1.0/stationratio:
                            st.remove(st.select(id=tr.id)[0])
                        else:
                            st.select(id=tr.id)[0].data = tr.data/40
                else:
                    for tr in [st[i] for i in component_2_indices]:
                        if mratio>stationratio:
                            st.remove(st.select(id=tr.id)[0])                        
                        else:  
                            st.select(id=tr.id)[0].data = tr.data/40
