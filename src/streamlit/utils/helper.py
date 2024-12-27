from scipy.signal import find_peaks
import pandas as pd

# Function to find the Top-N highest and lowest peaks in a single signal
def find_top_n_peaks(signal, n=5):
    """
    find the n peaks in the data
    """
    # Find highest peaks
    high_peaks, _ = find_peaks(signal)
    high_peaks_values = signal[high_peaks]
    top_high_peaks = sorted(zip(high_peaks, high_peaks_values), key=lambda x: -x[1])[:n]

    # Find lowest peaks 
    low_peaks, _ = find_peaks(signal * -1) # -1 means invert the signal to find the lowest peaks
    low_peaks_values = signal[low_peaks]
    top_low_peaks = sorted(zip(low_peaks, low_peaks_values), key=lambda x: x[1])[:n]

    return top_high_peaks, top_low_peaks

def find_peak_one_row(data) -> int:
    """
    ### find the peak which take the whole ecg signal 
    ### return the index 
    """
    high_peaks, _ = find_peaks(data)
    # as we need about 75 - 125 signal to identify the whole ecg period
    # in this case the maximum R wave may appear at first 6-10 , or 8-15
    # based on that we can conclude the valid peak may exists in range of [6, 160 ]
    valid_high_peaks = [ value for value in high_peaks if 6<=value<=160]
    high_peaks_values = data[high_peaks]
    high_peaks_values = data[ valid_high_peaks if len(valid_high_peaks)> 0 else high_peaks   ]
    top_high_peaks = sorted(zip(high_peaks, high_peaks_values), key=lambda x: -x[1])[:1]

    return top_high_peaks[0][0]

def shift_row(data) :
    """
    shift one row of data to align to the center of all dataset which is 87
    """
    center = 87 # default center
    peak = find_peak_one_row(data)
    shift = center - peak
    shifted_array = [0] * len(data)
    for i in range(len(data)):
        new_index = i + shift
        if 0 <= new_index < len(data):  # Ensure new index is within bounds
            shifted_array[new_index] = data[i]
            
    # return pd.DataFrame(shifted_array, columns=[f'c_{i}' for i in range(len(data))])
    return pd.DataFrame(shifted_array)