#All wave extraction, 7/4/2025, Basic Version
#Another new version edited on 8/29/2025 (Version 8)
import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import butter, filtfilt, sosfiltfilt, resample, iirnotch, find_peaks
from pathlib import Path
from biosppy.signals import ecg as ECG
import neurokit2 as nk 

class HeaParser:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def _get_hea_path(self, subject_id, study_id):
        return os.path.join(
            self.base_dir,
            "p" + subject_id[:4],
            "p" + subject_id,
            "s" + study_id,
            study_id + ".hea",
        )
    
    def _check_heapath(self, hea_path):
        if not os.path.exists(hea_path):
            raise FileNotFoundError(f"Not found: {hea_path}")
        return hea_path.replace(".hea", "")
    
    def _notch_filter(self, signal, fs, freq=60.0, q=30.0):
        nyq = 0.5 * fs
        w0 = freq / nyq
        b, a = iirnotch(w0, q)
        return filtfilt(b, a, signal, axis=0)

    def _butter_filter(self, signal, fs, lowcut=1, highcut=40.0):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(4, [low, high], btype="band", output="sos")
        return sosfiltfilt(sos, signal, axis=0)

    def _normalize(self, signal):
        mean = np.mean(signal, axis=0, keepdims=True)
        std = np.std(signal, axis=0, keepdims=True)
        return (signal - mean) / (std + 1e-8)

    def _resample(self, signal, old_fs, new_fs):
        n_samples = int(signal.shape[0] * new_fs / old_fs)
        return resample(signal, n_samples, axis=0)

    def _align_to_peak(self, record, target_duration=7.0):
        fs = record.fs
        anchor = int(fs * 0.42)  
        _end = int(fs * (0.42 + 1.5))
        target_samples = int(fs * target_duration)
        signal_ch2 = record.p_signal[anchor:_end, 1]
        peaks, _ = find_peaks(signal_ch2, height=2.5)

        if len(peaks) == 0:
            warnings.warn("No peak > 2.5 found, using any peak")
            peaks, _ = find_peaks(signal_ch2)

        peak_idx = anchor + (peaks[0] if len(peaks) else 0)
        start_idx = peak_idx - anchor
        end_idx = start_idx + target_samples

        record.p_signal = record.p_signal[start_idx:end_idx]
        return record
    
    def parse(
        self,
        subject_id,
        study_id,
        butter=True,
        notch=True,
        normalize=True,
        resample_fs=128,
        align_peak=True,
    ):
        hea_path = self._get_hea_path(subject_id, study_id)
        record_name = self._check_heapath(hea_path)
        record = wfdb.rdrecord(record_name, sampfrom=250, sampto=4750)

        fs = record.fs
        if notch:
            record.p_signal = self._notch_filter(record.p_signal, fs)
        if butter:
            record.p_signal = self._butter_filter(record.p_signal, fs)
        if normalize:
            record.p_signal = self._normalize(record.p_signal)
        if resample_fs and fs != resample_fs:
            record.p_signal = self._resample(record.p_signal, fs, resample_fs)
            record.fs = resample_fs
        if align_peak:
            record = self._align_to_peak(record)
        if normalize:
            record.p_signal = self._normalize(record.p_signal)

        return record

def rpeak_detection(ecg_data, sampling_rate=125.):
    rpeaks, = ECG.hamilton_segmenter(
        signal=ecg_data, sampling_rate=sampling_rate)
    rpeaks, = ECG.correct_rpeaks(
        signal=ecg_data,
        rpeaks=rpeaks,
        sampling_rate=sampling_rate,
        tol=0.05)
    return rpeaks

def detect_qrs_complex(ecg_signal, r_peaks, fs=125):
    q_peaks = []
    s_peaks = []
    
    for r_peak in r_peaks:
        # Q Detection Window (R-40ms)
        q_start = max(0, r_peak - int(0.04 * fs))
        q_end = r_peak
        q_region = ecg_signal[q_start:q_end]
        
        if len(q_region) > 0:
            # Detect Q wave (R- lowest point)
            q_idx = np.argmin(q_region)
            q_peak = q_start + q_idx
            q_peaks.append(q_peak)
        
        # S Detection Window (R+40ms)
        s_start = r_peak
        s_end = min(len(ecg_signal)-1, r_peak + int(0.04 * fs))
        s_region = ecg_signal[s_start:s_end]
        
        if len(s_region) > 0:
            # S Locate (R+ Lowest point)
            s_idx = np.argmin(s_region)
            s_peak = s_start + s_idx
            s_peaks.append(s_peak)
    
    return np.array(q_peaks), np.array(s_peaks)

def detect_t_p_peaks(ecg_signal, r_peaks, s_peaks, fs=125, t_min=60, t_max=400):
    t_peaks = []
    p_peaks = []

    n = min(len(r_peaks), len(s_peaks))
    if n == 0:
        return np.array(t_peaks), np.array(p_peaks)
    
    for i in range(n):
        r_peak = r_peaks[i]
        s_peak = s_peaks[i]
        
        # T wave detection window (60-400ms after S)
        t_min_samples = int(t_min * fs / 1000)
        t_max_samples = int(t_max * fs / 1000)
        
        # T wave detection window
        t_start = s_peak + t_min_samples
        t_end = s_peak + t_max_samples
        
        if t_start >= len(ecg_signal):
            continue
        if t_end >= len(ecg_signal):
            t_end = len(ecg_signal) - 1
        if t_end <= t_start:
            continue  
            
        t_region = ecg_signal[t_start:t_end]
        
        if len(t_region) > 0:
            t_peak_index = find_t_peak_with_gradient(t_region, fs)
            if t_peak_index >= 0: 
                t_peak = t_start + t_peak_index
                t_peaks.append(t_peak)
    
        # P wave detection window (50-300ms before R)
        p_start = max(0, r_peak - int(0.3 * fs)) 
        p_end = r_peak - int(0.05 * fs) 
        
        if p_end > p_start and p_start < len(ecg_signal) and p_end < len(ecg_signal):
            p_region = ecg_signal[p_start:p_end]
            if len(p_region) > 0:
                # P Locate (max point)
                p_idx = np.argmax(p_region)
                p_peak = p_start + p_idx
                
                # Verify P wave amplitude
                baseline = np.mean(p_region[:min(5, len(p_region))])  
                if ecg_signal[p_peak] - baseline > 0.05:  
                    p_peaks.append(p_peak)
                
    return np.array(t_peaks), np.array(p_peaks)
     

def find_t_peak_with_gradient(ecg_segment, fs, min_amplitude=0.1):
    """
    #​robust T-wave peak detection algorithm​
    1. Use gradient to find slope change points
    2. Select the point with the largest amplitude change from baseline
    3. Improve: add 50-500 ms window limitation. Avoid T wave disappear and mismarked P-peak
    4. Improve: add abs val comparison to handle inverted T-waves
    5. TODO: need extra data to validate the accuracy of the algorithm
    """
    # Calculate baseline using the first 20ms
    baseline_window = min(int(0.02 * fs), len(ecg_segment))
    baseline = np.mean(ecg_segment[:baseline_window]) if baseline_window > 0 else 0
    
    # Get gradient
    gradient = np.gradient(ecg_segment)
    
    # Gradient sign changes
    sign_changes = np.where(np.diff(np.sign(gradient)))[0]
    
    # Choose candidate peaks based on sign changes
    candidate_peaks = sign_changes
    candidate_amplitudes = np.abs(ecg_segment[candidate_peaks] - baseline)
    
    # Filter candidates by minimum amplitude
    valid_indices = np.where(candidate_amplitudes >= min_amplitude)[0]
    
    if len(valid_indices) == 0:
        # If no valid candidate, return the point with max amplitude if it meets min_amplitude
        max_idx = np.argmax(np.abs(ecg_segment - baseline))
        if np.abs(ecg_segment[max_idx] - baseline) >= min_amplitude:
            return max_idx
        return -1
    
    # Choose the candidate with the highest amplitude change
    best_index = valid_indices[np.argmax(candidate_amplitudes[valid_indices])]
    best_candidate = candidate_peaks[best_index]
    
    return best_candidate


def plot_ecg_with_peaks(ecg_signal, r_peaks, t_peaks, fs, title="ECG with Detected Peaks"):
    plt.figure(figsize=(15, 6))
    time = np.arange(len(ecg_signal)) / fs
    plt.plot(time, ecg_signal, label='ECG Signal', linewidth=1.5)
    
    plt.scatter(time[r_peaks], ecg_signal[r_peaks], 
                color='red', marker='v', s=80, label='R Peaks')
    
    if len(t_peaks) > 0:
        plt.scatter(time[t_peaks], ecg_signal[t_peaks], 
                    color='green', marker='^', s=80, label='T Peaks')
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_ecg_with_all_peaks(ecg_signal, r_peaks, q_peaks, s_peaks, t_peaks, p_peaks, fs, 
                           title="ECG with All Detected Peaks"):
    plt.figure(figsize=(15, 8))
    time = np.arange(len(ecg_signal)) / fs
    
    plt.plot(time, ecg_signal, label='ECG Signal', linewidth=1.5, alpha=0.7)
    
    if len(p_peaks) > 0:
        plt.scatter(time[p_peaks], ecg_signal[p_peaks], 
                    color='magenta', marker='*', s=100, label='P Peaks')
    
    if len(q_peaks) > 0:
        plt.scatter(time[q_peaks], ecg_signal[q_peaks], 
                    color='purple', marker='x', s=100, label='Q Peaks')
    
    plt.scatter(time[r_peaks], ecg_signal[r_peaks], 
                color='red', marker='v', s=100, label='R Peaks')
    
    if len(s_peaks) > 0:
        plt.scatter(time[s_peaks], ecg_signal[s_peaks], 
                    color='blue', marker='x', s=100, label='S Peaks')
    
    if len(t_peaks) > 0:
        plt.scatter(time[t_peaks], ecg_signal[t_peaks], 
                    color='green', marker='^', s=100, label='T Peaks')
    
    if len(p_peaks) > 0:
        for i, p in enumerate(p_peaks):
            next_rs = [r for r in r_peaks if r > p]
            if next_rs:
                r = next_rs[0]
                plt.plot([time[p], time[r]], 
                         [ecg_signal[p], ecg_signal[r]], 
                         'm--', alpha=0.5, linewidth=1)
    
    if len(q_peaks) > 0 and len(s_peaks) > 0:
        for i in range(min(len(q_peaks), len(r_peaks), len(s_peaks))):
            q = q_peaks[i]
            r = r_peaks[i]
            s = s_peaks[i]
            
            plt.plot([time[q], time[r], time[s]], 
                     [ecg_signal[q], ecg_signal[r], ecg_signal[s]], 
                     'm-', linewidth=2, alpha=0.7)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    BASE_DIR = r"C:\Users\DELL\Desktop\DataECG"
    parser = HeaParser(base_dir=BASE_DIR)
      
    subject_id = "10003731"
    study_id = "41861712"
    record = parser.parse(subject_id, study_id)
    
    # Lead II
    ecg_signal = record.p_signal[:, 1]
    fs = record.fs
    
    print(f"ECG Length: {len(ecg_signal)} fs points, fs: {fs} Hz")
    
    # Detect R peaks
    print("Detecting R peaks...")
    rpeaks = rpeak_detection(ecg_signal, sampling_rate=fs)
    print(f"Detected {len(rpeaks)} R peaks")
    
    # Detect Q and S peaks
    print("Detecting Q and S peaks...")
    q_peaks, s_peaks = detect_qrs_complex(ecg_signal, rpeaks, fs=fs)
    print(f"Detected {len(q_peaks)} Q peak, {len(s_peaks)} S peak")

    # Detect T and P peaks
    print("Detecting T and P peaks...")
    t_peaks, p_peaks = detect_t_p_peaks(ecg_signal, rpeaks, s_peaks, fs=fs)
    print(f"Detected {len(t_peaks)} T wave")
    print(f"Detected {len(p_peaks)} P wave")
    
    
    # Visiualize (R and T waves only)
    # plot_ecg_with_peaks(ecg_signal, rpeaks, t_peaks, fs, 
    #                    title=f"ECG Record {subject_id}/{study_id} (R & T Waves)")
    
    # Visualize (All waves)
    plot_ecg_with_all_peaks(ecg_signal, rpeaks, q_peaks, s_peaks, t_peaks, p_peaks, fs, 
                           title=f"ECG Record {subject_id}/{study_id} (All Waves)")
    
    # Extract features
    if len(rpeaks) > 1 and len(t_peaks) > 0:
        # Calculate R-T intervals
        rt_intervals = [((t - r)/fs)*1000 for r, t in zip(rpeaks[:len(t_peaks)], t_peaks)]
        print(f"Avg R-T intervals: {np.mean(rt_intervals):.1f} ms")
        
        # Calculate R-T amplitudes
        amplitudes = [ecg_signal[r] - ecg_signal[t] for r, t in zip(rpeaks[:len(t_peaks)], t_peaks)]
        print(f"Avg R-T amplitudes: {np.mean(amplitudes):.3f} mV")
    
    if len(q_peaks) > 0 and len(s_peaks) > 0:
        qrs_durations = [(s - q) / fs * 1000 for q, s in zip(q_peaks, s_peaks)]
        print(f"Avg QRS lasting time: {np.mean(qrs_durations):.1f} ms")
    
    if len(p_peaks) > 0 and len(rpeaks) > 0:
        pr_intervals = []
        for p in p_peaks:
            next_rs = [r for r in rpeaks if r > p]
            if next_rs:
                pr_interval = (next_rs[0] - p) / fs * 1000
                pr_intervals.append(pr_interval)
        
        if pr_intervals:
            print(f"Avg PR intervals: {np.mean(pr_intervals):.1f} ms")


