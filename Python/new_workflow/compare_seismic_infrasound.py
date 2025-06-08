import numpy as np
import matplotlib.pyplot as plt
from obspy.signal.cross_correlation import xcorr_max
from obspy.signal.util import next_pow_2
from scipy.stats import linregress


def analyze_seismo_acoustic_pair(tr_seis, tr_acou, title="Seismo-Acoustic Analysis",
                                 freqmin=0.1, freqmax=10.0,
                                 window_duration=60.0, sliding=True):
    """
    Compare infrasound and seismic Trace objects using time series, amplitude regression,
    spectral ratios, and cross-correlation (with optional sliding window).

    Parameters:
    -----------
    tr_seis : obspy.Trace
        Seismic trace (e.g., vertical ground motion)
    tr_acou : obspy.Trace
        Infrasound trace (e.g., barometric pressure)
    title : str
        Title for all plots
    freqmin, freqmax : float
        Bandpass filter limits in Hz
    window_duration : float
        Duration in seconds to trim both traces (from starttime)
    sliding : bool
        If True, compute sliding window cross-correlation
    """

    # === Trim to common window ===
    t0 = max(tr_seis.stats.starttime, tr_acou.stats.starttime)
    t1 = t0 + window_duration
    tr_seis = tr_seis.copy().trim(t0, t1)
    tr_acou = tr_acou.copy().trim(t0, t1)

    # === Preprocess ===
    for tr in (tr_seis, tr_acou):
        tr.detrend("demean")
        tr.taper(0.05)
        tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

    # === Match lengths ===
    minlen = min(len(tr_seis), len(tr_acou))
    x = tr_acou.data[:minlen]
    y = tr_seis.data[:minlen]
    sr = tr_seis.stats.sampling_rate
    times = np.arange(minlen) / sr

    # === Plot time series ===
    plt.figure(figsize=(10, 4))
    plt.plot(times, y, label=tr_seis.stats.channel)
    plt.plot(times, x, label=tr_acou.stats.channel, alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(f"Time Series: {title}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # === Amplitude regression ===
    slope, intercept, r_value, _, _ = linregress(x, y)

    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, alpha=0.3, label="Samples")
    plt.plot(x, slope * x + intercept, color='red',
             label=f"Fit: y = {slope:.2f}x + {intercept:.2f}\nR² = {r_value**2:.3f}")
    plt.xlabel("Infrasound Amplitude")
    plt.ylabel("Seismic Amplitude")
    plt.title(f"Amplitude Scatter + Regression: {title}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # === Spectral ratio ===
    nfft = next_pow_2(minlen)
    freqs = np.fft.rfftfreq(nfft, d=1/sr)
    fft_seis = np.fft.rfft(y, n=nfft)
    fft_acou = np.fft.rfft(x, n=nfft)
    ratio_db = 20 * np.log10(np.abs(fft_seis) / np.abs(fft_acou))
    ratio_db[np.isinf(ratio_db)] = np.nan

    plt.figure(figsize=(10, 4))
    plt.semilogx(freqs, ratio_db)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Seismic / Infrasound Amplitude Ratio (dB)")
    plt.title(f"Spectral Ratio: {title}")
    plt.grid(which="both")
    plt.tight_layout()
    plt.show()

    # === Cross-correlation ===
    cc_max, lag_samples = xcorr_max(x, y, shift=minlen // 2)
    lag_time = lag_samples / sr
    print(f"Global Max Cross-Correlation = {cc_max:.3f}, Time Lag = {lag_time:.3f} s")

    if sliding:
        win_len = 5.0  # seconds
        step = 1.0     # seconds
        nwin = int(win_len * sr)
        nstep = int(step * sr)

        times_slide = []
        lags = []
        corrs = []

        for i in range(0, minlen - nwin, nstep):
            w1 = x[i:i + nwin]
            w2 = y[i:i + nwin]
            cc, lag = xcorr_max(w1, w2, shift=nwin // 2)
            times_slide.append(i / sr)
            lags.append(lag / sr)
            corrs.append(cc)

        # Plot sliding results
        plt.figure(figsize=(10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(times_slide, corrs)
        plt.ylabel("Max Corr")
        plt.title(f"Sliding Cross-Correlation: {title}")
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(times_slide, lags)
        plt.xlabel("Time (s)")
        plt.ylabel("Lag (s)")
        plt.grid()

        plt.tight_layout()
        plt.show()


from obspy.core import Stream, UTCDateTime


def main(stream: Stream, starttime: UTCDateTime, endtime: UTCDateTime):
    """
    Analyze all seismo-acoustic trace pairs in the given stream between starttime and endtime.

    Parameters:
    -----------
    stream : obspy.Stream
        The input Stream containing multiple stations and channels
    starttime : obspy.UTCDateTime
        Start of analysis window
    endtime : obspy.UTCDateTime
        End of analysis window
    """

    print(f"Analyzing seismo-acoustic pairs from {starttime} to {endtime}")
    stream = stream.copy().trim(starttime, endtime)

    stations = set(tr.stats.station for tr in stream)
    print(f"Found {len(stations)} stations: {stations}")

    for station in stations:
        st_station = stream.select(station=station)
        seismic_traces = st_station.select(channel="*Z")  # e.g., BHZ, HHZ, EHZ
        infrasound_traces = st_station.select(channel="BD*", "DF*", "LD*")  # customize as needed

        if not seismic_traces or not infrasound_traces:
            print(f"Skipping station {station}: missing seismic or infrasound channel.")
            continue

        for tr_seis in seismic_traces:
            for tr_acou in infrasound_traces:
                print(f"\n>>> Station {station}: {tr_seis.id} vs {tr_acou.id}")
                try:
                    analyze_seismo_acoustic_pair(
                        tr_seis,
                        tr_acou,
                        title=f"{station}: {tr_seis.stats.channel} vs {tr_acou.stats.channel}"
                    )
                except Exception as e:
                    print(f"Failed analysis for {tr_seis.id} vs {tr_acou.id}: {e}")

if __name__ == "__main__":
    from obspy import read, UTCDateTime

    # Load your full dataset
    st = read("launch_data_all_stations.mseed")

    # Define time window (e.g., around a rocket launch)
    t0 = UTCDateTime("2024-06-05T14:00:00")
    t1 = t0 + 90  # 90-second window

    # Run the analysis
    main(st, starttime=t0, endtime=t1)
