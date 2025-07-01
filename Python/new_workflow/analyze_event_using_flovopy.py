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
"""
from flovopy.core.usf import NRL2inventory

def get_stationXML_inventory(xmlfile='KSC.xml'):
    preseconds=120
    eventseconds=120
    postseconds=120
    taperseconds=600

    # need to add a detector for sonic booms too!
    # 2022 responses
    ondate = UTCDateTime(2016, 2, 24)
    offdate = UTCDateTime(2022,12,5)

    if os.path.isfile(xmlfile):
        inv = read_inventory(xmlfile)
    else:

        # seismic channels
        invs1 = NRL2inventory('FL', 'S39A1', '00', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        invs2 = NRL2inventory('FL', 'S39A2', '00', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        invs3 = NRL2inventory('FL', 'S39A3', '00', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        invs4 = NRL2inventory('FL', 'BCHH3', '00', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        invs5 = NRL2inventory('FL', 'BCHH4', '10', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        seismicinv = invs1 + invs2 + invs3 + invs4 + invs5

        # infrasound
        date_BCHH3_change = UTCDateTime(2022,5,26) # looking at files, it seems we actually have no data for BCHH3 after this - we just called it BCHH4
        invi1 = NRL2inventory('FL', 'S39A1', '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)
        invi2 = NRL2inventory('FL', 'S39A2', '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)
        invi3 = NRL2inventory('FL', 'S39A3', '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)
        invi4 = NRL2inventory('FL', 'BCHH2', '10', ['HD4'], datalogger='Centaur', sensor='Chap', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        invi5 = NRL2inventory('FL', 'BCHH2', '10', ['HD5', 'HD6'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        invi6 = NRL2inventory('FL', 'BCHH2', '10', ['HD7', 'HD8', 'HD9'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)
        invi7 = NRL2inventory('FL', 'BCHH3', '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=date_BCHH3_change)
        invi8 = NRL2inventory('FL', 'BCHH3', '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=100.0, ondate=date_BCHH3_change, offdate=offdate)
        invi9 = NRL2inventory('FL', 'BCHH4', '00', ['HDF', 'HD2', 'HD3'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)

        infrainv = invi1 + invi2 + invi3 + invi4 + invi5 + invi6 + invi7 + invi8 + invi9


        for station_num in range(1,9,1):
            # 2018-2021
            station = f'BHP{station_num}'
            invpasscal = NRL2inventory('1R', station, '', ['EH1', 'EH2', 'EHZ'], datalogger='RT130', sensor='L-22', fsamp=100.0, ondate=ondate, offdate=offdate)
            seismicinv = seismicinv + invpasscal
        for station in ['FIREP', 'TANKP']:
            invpasscal = NRL2inventory('1R', station, '', ['EH1', 'EH2', 'EHZ'], datalogger='RT130', sensor='L-22', fsamp=100.0, ondate=ondate, offdate=offdate)
            seismicinv = seismicinv + invpasscal

        inv0 = NRL2inventory('FL', 'BCHH1', '0', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        seismicinv = seismicinv + inv0
        inv1 = NRL2inventory('FL', 'BCHH1', '0', ['HD1', 'HD2', 'HD3'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        infrainv = infrainv + inv1

        inv2 = NRL2inventory('FL', 'BCHH', '00', ['GHZ', 'GHN', 'GHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=2000.0, ondate=ondate, offdate=offdate)
        seismicinv = seismicinv + inv2
        inv3 = NRL2inventory('FL', 'BCHH', '10', ['GD1', 'GD2', 'GD3'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=2000.0, ondate=ondate, offdate=offdate)
        infrainv = infrainv + inv3

        for station in ['BCHH', 'FIRE', 'TANK']:
            inv4 = NRL2inventory('FL', station, '00', ['DHZ', 'DHN', 'DHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=250.0, ondate=ondate, offdate=offdate)
            seismicinv = seismicinv + inv4
            inv5 = NRL2inventory('FL', station, '10', ['DD1', 'DD2', 'DD3'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=250.0, ondate=ondate, offdate=offdate)
            infrainv = infrainv + inv5

        inv6 = NRL2inventory('FL', 'BCHH2', '10', ['HD4', 'HD5', 'HD6', 'HD7', 'HD8', 'HD9'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=250.0, ondate=ondate, offdate=offdate)
        infrainv = infrainv + inv6


        inv = seismicinv + infrainv

        inv.write(xmlfile, format='STATIONXML')
        return inv
    
"""
from obspy import UTCDateTime, read_inventory
from flovopy.core.usf import NRL2inventory, apply_coordinates_from_csv, merge_duplicate_stations_and_patch_site, write_inventory_as_resp
import os

def get_stationXML_inventory(
    xmlfile='KSC.xml',
    seedfile='KSC.dataless',
    respdir='RESP/',
    coord_csv='station_coordinates.csv',
    overwrite=False
):
    from obspy import Inventory

    if os.path.isfile(xmlfile) and not overwrite:
        inv = read_inventory(xmlfile)
        print(f"[INFO] Loaded existing StationXML: {xmlfile}")
    else:
        print("[INFO] Creating new inventory from USF definitions...")
        ondate = UTCDateTime(2016, 2, 24)
        offdate = UTCDateTime(2022, 12, 5)
        date_BCHH3_change = UTCDateTime(2022, 5, 26)

        seismicinv = Inventory()
        infrainv = Inventory()

        # === Seismic inventory ===
        for code in ['S39A1', 'S39A2', 'S39A3']:
            seismicinv += NRL2inventory('FL', code, '00', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        seismicinv += NRL2inventory('FL', 'BCHH3', '00', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        seismicinv += NRL2inventory('FL', 'BCHH4', '10', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)

        for station_num in range(1, 9):
            seismicinv += NRL2inventory('1R', f'BHP{station_num}', '', ['EH1', 'EH2', 'EHZ'], datalogger='RT130', sensor='L-22', fsamp=100.0, ondate=ondate, offdate=offdate)
        for station in ['FIREP', 'TANKP']:
            seismicinv += NRL2inventory('1R', station, '', ['EH1', 'EH2', 'EHZ'], datalogger='RT130', sensor='L-22', fsamp=100.0, ondate=ondate, offdate=offdate)
        seismicinv += NRL2inventory('FL', 'BCHH1', '0', ['HHZ', 'HHN', 'HHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        seismicinv += NRL2inventory('FL', 'BCHH', '00', ['GHZ', 'GHN', 'GHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=2000.0, ondate=ondate, offdate=offdate)
        for station in ['BCHH', 'FIRE', 'TANK']:
            seismicinv += NRL2inventory('FL', station, '00', ['DHZ', 'DHN', 'DHE'], datalogger='Centaur', sensor='TCP', Vpp=40, fsamp=250.0, ondate=ondate, offdate=offdate)

        # === Infrasound inventory ===
        for code in ['S39A1', 'S39A2', 'S39A3']:
            infrainv += NRL2inventory('FL', code, '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)

        infrainv += NRL2inventory('FL', 'BCHH2', '10', ['HD4'], datalogger='Centaur', sensor='Chap', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        infrainv += NRL2inventory('FL', 'BCHH2', '10', ['HD5', 'HD6'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        infrainv += NRL2inventory('FL', 'BCHH2', '10', ['HD7', 'HD8', 'HD9'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)
        infrainv += NRL2inventory('FL', 'BCHH3', '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=date_BCHH3_change)
        infrainv += NRL2inventory('FL', 'BCHH3', '10', ['HDF'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=100.0, ondate=date_BCHH3_change, offdate=offdate)
        infrainv += NRL2inventory('FL', 'BCHH4', '00', ['HDF', 'HD2', 'HD3'], datalogger='Centaur', sensor='infraBSU', Vpp=1, fsamp=100.0, ondate=ondate, offdate=offdate)
        infrainv += NRL2inventory('FL', 'BCHH1', '0', ['HD1', 'HD2', 'HD3'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=100.0, ondate=ondate, offdate=offdate)
        infrainv += NRL2inventory('FL', 'BCHH', '10', ['GD1', 'GD2', 'GD3'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=2000.0, ondate=ondate, offdate=offdate)
        for station in ['BCHH', 'FIRE', 'TANK']:
            infrainv += NRL2inventory('FL', station, '10', ['DD1', 'DD2', 'DD3'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=250.0, ondate=ondate, offdate=offdate)
        infrainv += NRL2inventory('FL', 'BCHH2', '10', ['HD4', 'HD5', 'HD6', 'HD7', 'HD8', 'HD9'], datalogger='Centaur', sensor='infraBSU', Vpp=40, fsamp=250.0, ondate=ondate, offdate=offdate)

        # Combine
        inv = seismicinv + infrainv

        # Apply coordinates from CSV
        if os.path.isfile(coord_csv):
            apply_coordinates_from_csv(inv, coord_csv)

        # Merge + site patch
        inv = merge_duplicate_stations_and_patch_site(inv)

        # Save StationXML
        inv.write(xmlfile, format='STATIONXML', validate=True)
        print(f"[OK] Wrote StationXML to {xmlfile}")

        # Write RESP
        write_inventory_as_resp(inv, seed_tempfile=seedfile, resp_outdir=respdir)

    return inv



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

def air_to_ground_coupling(seismic_trace, infrasound_trace, water_level=0.01, plot=False):
    
    npts = min(len(seismic_trace.data), len(infrasound_trace.data))
    dt = seismic_trace.stats.delta

    st1 = seismic_trace.copy()
    st2 = infrasound_trace.copy()


    s = st1.data
    i = st2.data

    S = np.fft.rfft(s)
    I = np.fft.rfft(i)

    I_mag = np.abs(I)
    I_stabilized = np.where(I_mag < water_level, water_level, I_mag)
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
    stream = stream.copy().trim(starttime, endtime)
    preprocess_stream(stream, inv=inventory, outputType="VEL", bool_clean=True)
    remove_low_quality_traces(stream, quality_threshold=4.0, verbose=True)
    est = EnhancedStream(stream=stream)
    print(est)
    est.ampengfft()
    print(est)
    input('ENTER to continue')

    stations = set(tr.stats.station for tr in stream)
    all_metrics = []
    trace_metrics = []

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
        print()

    for station in stations:
        st_station = est.select(station=station)
        traces = st_station.select(channel="HH*") + st_station.select(channel="HD*")

        for i, tr1 in enumerate(traces):
            for j, tr2 in enumerate(traces):
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

                    if kind == "seis-acou":
                        h_trace = air_to_ground_coupling(tr1, tr2)
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
    eventtime = "2022-11-01T13:41:00"
    t0 = UTCDateTime(eventtime)
    t1 = t0 + 90

    stS = read(f"/data/KSC/EROSION/EVENTS/{eventtime}/seismic.mseed")
    stI = read(f"/data/KSC/EROSION/EVENTS/{eventtime}/infrasound.mseed")
    inventory = get_stationXML_inventory(
        xmlfile='/data/KSC/station_metadata/KSC.xml',
        seedfile='/data/KSC/station_metadata/KSC.dataless',
        respdir='/data/KSC/station_metadata/RESP/',
        coord_csv='/data/KSC/station_metadata/station_coordinates.csv',
        )
    st = stS + stI

    df_results, df_trace_metrics = main(st, t0, t1, inventory)
