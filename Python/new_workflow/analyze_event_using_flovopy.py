import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime, Trace, Stream, read_inventory
from obspy.signal.cross_correlation import xcorr_max, correlate
from obspy.signal.util import next_pow_2
from scipy.stats import linregress
import os
from flovopy.core.preprocessing import preprocess_stream, remove_low_quality_traces
from flovopy.core.enhanced import EnhancedStream

#from obspy import UTCDateTime, read_inventory
from flovopy.core.usf import get_stationXML_inventory, inventory2dataless_and_resp





def summarize_coupling_function(h_trace, energy_percent=0.9):
    h = h_trace.data
    t = h_trace.times()
    abs_h = np.abs(h)
    peak_amp = np.max(abs_h)
    lag_time = t[np.argmax(abs_h)]

    energy = h**2
    cumulative_energy = np.cumsum(energy)
    total_energy = cumulative_energy[-1]
    start_idx = np.argmax(cumulative_energy >= (1 - energy_percent) * total_energy)
    end_idx = np.argmax(cumulative_energy >= energy_percent * total_energy)
    duration = t[end_idx] - t[start_idx]

    return peak_amp, lag_time, duration

def cross_spectral_peak(seis, infra):
    x = seis.data
    y = infra.data
    cc = correlate(x, y, shift=len(x)//2)
    X = np.fft.rfft(cc)
    freqs = np.fft.rfftfreq(len(cc), d=seis.stats.delta)
    spectrum = np.abs(X)
    peak_freq = freqs[np.argmax(spectrum)]
    return peak_freq

def trace_basic_metrics(tr):
    data = tr.data
    abs_data = np.abs(data)
    peak_amp = np.max(abs_data)
    peak_index = np.argmax(abs_data)
    peak_time = tr.stats.starttime + peak_index / tr.stats.sampling_rate

    nfft = next_pow_2(len(data))
    freqs = np.fft.rfftfreq(nfft, d=tr.stats.delta)
    spectrum = np.abs(np.fft.rfft(data, n=nfft))
    dom_freq = freqs[np.argmax(spectrum)]

    return {
        "trace_id": tr.id,
        "station": tr.stats.station,
        "channel": tr.stats.channel,
        "peak_amplitude": peak_amp,
        "time_of_peak": str(peak_time),
        "dominant_frequency": dom_freq
    }

def air_to_ground_coupling(seismic_trace, infrasound_trace, water_level=0.01, taper_fraction = 0.05, plot=False):
    
    npts = min(len(seismic_trace.data), len(infrasound_trace.data))
    dt = seismic_trace.stats.delta

    st1 = seismic_trace.copy()
    st2 = infrasound_trace.copy()


    s = st1.data
    i = st2.data

    S = np.fft.rfft(s)
    I = np.fft.rfft(i)

    I_mag = np.abs(I)


    # Optionally stabilize I first to prevent division by zero
    I_stabilized = np.where(np.abs(I) < 1e-10, 1e-10, I)

    # Clean non-finite values from I and S
    I_stabilized = np.nan_to_num(I_stabilized, nan=1e-10, posinf=1e10, neginf=-1e10)
    S = np.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    # Now safe to compute
    H = S / (I_stabilized * np.exp(1j * np.angle(I)))


    h = np.fft.irfft(H, n=npts)

    h_trace = Trace(data=h.astype(np.float32))
    h_trace.stats = seismic_trace.stats.copy()
    h_trace.stats.channel = f"H_{seismic_trace.stats.channel}"

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(h_trace.times(), h_trace.data)
        plt.title(f"Impulse Response: {seismic_trace.id} vs {infrasound_trace.id}")
        plt.xlabel("Time (s")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.tight_layout()
        plt.show()

    return h_trace



def main(stream, starttime, endtime, inventory):
    stream.write('/home/thompsong/Dropbox/launchraw.mseed')
    stream = stream.copy().trim(starttime, endtime)
    stream.write('/home/thompsong/Dropbox/launchrawtrimmed.mseed')
    preprocess_stream(stream, inv=inventory, filterType='highpass', freq=[0.1, 100.0], 
                      outputType="DEF", bool_clean=True, taperFraction=0.05, bool_detrend=True)
    stream.write('/home/thompsong/Dropbox/launchpreprocessed.mseed')
    stream.plot(equal_scale=False)
    remove_low_quality_traces(stream, quality_threshold=4.0, verbose=True)
    est = EnhancedStream(stream=stream)
    print(est)
    stream.write('/home/thompsong/Dropbox/launchenhanced.pkl', format='PICKLE')
    est.ampengfft()
    stream.write('/home/thompsong/Dropbox/launchAEF.pkl', format='PICKLE')
    print(est)
    #input('ENTER to continue')



    for tr in est:
        try:
            m = getattr(tr.stats, 'metrics', {})
            trace_metrics.append({
                "trace_id": tr.id,
                "station": tr.stats.station,
                "channel": tr.stats.channel,
                "peak_amplitude": m.get("peakamp"),
                "time_of_peak": m.get("peaktime"),
                "dominant_frequency": m.get("peakf")
            })
        except Exception as e:
            print(f"Trace metrics failed for {tr.id}: {e}")
        print(tr.stats)
        #tr.plot()
        print()
    stream.write('/home/thompsong/Dropbox/launchAEF2.pkl', format='PICKLE')
    return
    stations = set(tr.stats.station for tr in stream)
    all_metrics = []
    trace_metrics = []

    for station in stations:
        st_station = est.select(station=station)
        traces = st_station.select(channel="HH*") + st_station.select(channel="HD*")

        for i, tr1 in enumerate(traces):
            for j, tr2 in enumerate(traces):
                these_traces = Stream(traces=[tr1, tr2])
                these_traces.plot(equal_scale=False)
                try:
                    trace1_id = tr1.id
                    trace2_id = tr2.id
                    kind = "auto" if i == j else (
                        "seis-seis" if tr1.stats.channel.startswith("HH") and tr2.stats.channel.startswith("HH") else
                        "acou-acou" if tr1.stats.channel.startswith("HD") and tr2.stats.channel.startswith("HD") else
                        "seis-acou"
                    )

                    metrics = {
                        "station": station,
                        "trace1_id": trace1_id,
                        "trace2_id": trace2_id,
                        "type": kind
                    }


                    #if kind == "seis-acou":
                    if kind:
                        h_trace = air_to_ground_coupling(tr1, tr2, plot=True)
                        peak_amp, lag, duration = summarize_coupling_function(h_trace)
                        peak_freq = cross_spectral_peak(tr1, tr2)

                        minlen = min(len(tr1), len(tr2))
                        sr = tr1.stats.sampling_rate
                        x = tr1.data[:minlen]
                        y = tr2.data[:minlen]

                        slope, intercept, r_value, _, _ = linregress(x, y)
                        cc = correlate(x, y, shift=minlen // 2)
                        cc_max, lag_samples = xcorr_max(cc)
                        lag_time = lag_samples / sr

                        metrics.update({
                            "r_value": r_value,
                            "cc_max": cc_max,
                            "lag_s": lag_time,
                            "coupling_peak_amp": peak_amp,
                            "coupling_lag_s": lag,
                            "coupling_duration_s": duration,
                            "peak_crosscorr_freq_Hz": peak_freq
                        })

                    all_metrics.append(metrics)

                except Exception as e:
                    print(f"Failed {tr1.id} vs {tr2.id}: {e}")

    df = pd.DataFrame(all_metrics)
    df_traces = pd.DataFrame(trace_metrics)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/coupling_metrics.csv", index=False)
    df_traces.to_csv("output/trace_metrics.csv", index=False)
    print("Saved results to output/coupling_metrics.csv and output/trace_metrics.csv")
    return df, df_traces

if __name__ == "__main__":
    xmlfile = '/data/station_metadata/KSC.xml'
    #dataless = '/data/station_metadata/KSC.dataless'
    respdir = '/data/station_metadata/RESP'   
    if os.path.isfile(xmlfile):
        inv = read_inventory(xmlfile)
        print(f"[INFO] Loaded existing StationXML: {xmlfile}")
    else:
        print("[INFO] Creating new inventory from USF definitions...")
        inv = get_stationXML_inventory(xmlfile=xmlfile, overwrite=True)
        inventory2dataless_and_resp(inv, output_dir=respdir,
                                stationxml_seed_converter_jar="/home/thompsong/stationxml-seed-converter.jar")
    eventtime = "2022-11-01T13:41:00"
    t0 = UTCDateTime(eventtime) - 300
    t1 = t0 + 500

    stS = read(f"/data/KSC/EROSION/EVENTS/{eventtime}/seismic.mseed")
    stI = read(f"/data/KSC/EROSION/EVENTS/{eventtime}/infrasound.mseed")

    st = stS + stI

    df_results, df_trace_metrics = main(st, t0, t1, inv)
