import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.signal import freqz, lfilter 
import numpy as np 
from scipy.fft import fft, fftshift 
 
def Data_visualization (file_name :str, sheet_name: str): 
    #obtain the time and amplitude  
    amplitude, time = get_excel_data(file_name, sheet_name) 
 
    #plot the data 
    plt.figure() 
    plt.plot(time, amplitude, 'b', label='Ali: 1212171 \nKhalid: 1210618') 
    plt.xlabel('Time (seconds)') 
    plt.ylabel('Amplitude (mv)') 
    plt.title(sheet_name + " visualization") 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
def get_excel_data (file_name :str, sheet_name: str): 
    # read the excel file 
    data = pd.read_excel(file_name, sheet_name, header=1) # header=1 to ignore  the first row which contains column name 
    # obtain the required columns from the file  
    amplitude = data.iloc[:, 2] # the amplitude is thurd column 
    time = data.iloc[:, 1] # the real time is the second column 
    return amplitude.tolist(), time.tolist() 
 
def high_pass_filter (singal_name: str): 
    #defning the nominator and denominator of filter transfer function 
    numerator = [0] * 33 
    numerator[0], numerator[16], numerator[17], numerator[32] = -1/32, 1, -1, 1/32 
    denominator = [1, -1] 
    #define the filter 
    # detetmine the freq. respose 
    w, h = freqz(numerator, denominator, worN=np.linspace(-1*np.pi, 1 * np.pi, 2048)) 
    # plot the frequecy response  
    plt.figure() 
    # Magnitude response  
 
    plt.subplot(2, 1, 1) 
    plt.plot(w, abs(h), 'b', label='Ali: 1212171 \nKhalid: 1210618') 
    plt.title('Magnitude and Phase Response of the High Pass Filter') 
    plt.ylabel('Magnitude') 
    plt.legend() 
    plt.grid() 
 
    # Phase response 
    plt.subplot(2, 1, 2) 
    plt.plot(w, np.angle(h), 'b') 
    plt.xlabel('Frequency [radians / second]') 
    plt.ylabel('Phase [radians]') 
    plt.grid() 
 
    plt.show() 
     
    #applying filter to the signal 
    amplitude, time = get_excel_data('Data_ECG_raw.xlsx', singal_name) 
    filtered_signal = lfilter(numerator, denominator, amplitude) 
     
    #plot the filtered signal 
    plt.figure() 
    plt.plot(time, filtered_signal, 'b', label='Ali: 1212171 \nKhalid: 1210618') 
    plt.xlabel('Time (seconds)') 
    plt.ylabel('Amplitude (mv)') 
    plt.title('High Pass Filtered ' + singal_name) 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
def low_pass_filter (signal_name: str): 
    #defning the nominator and denominator of filter transfer function 
    numerator = [0] * 13 
    numerator[0], numerator[6], numerator[12] = 1, -2, 1 
    denominator = [1, -2, 1] 
 
    #define the filter 
    # detetmine the freq. respose 
    w, h = freqz(numerator, denominator, worN=np.linspace(-1*np.pi, 1 * np.pi, 2048)) 
    # plot the frequecy response  
    plt.figure() 
    # Magnitude response 
    plt.subplot(2, 1, 1) 
    plt.plot(w, abs(h), 'b', label='Ali: 1212171 \nKhalid: 1210618') 
 
 
    plt.title('Magnitude and Phase Response of the Low Pass Filter') 
    plt.ylabel('Magnitude') 
    plt.legend() 
    plt.grid() 
 
    # Phase response 
    plt.subplot(2, 1, 2) 
    plt.plot(w, np.angle(h), 'b') 
    plt.xlabel('Frequency [radians / second]') 
    plt.ylabel('Phase [radians]') 
    plt.grid() 
 
    plt.show() 
     
    #applying filter to the signal 
    amplitude, time = get_excel_data('Data_ECG_raw.xlsx', signal_name) 
    filtered_signal = lfilter(numerator, denominator, amplitude) 
     
    #plot the filtered signal 
    plt.figure() 
    plt.plot(time, filtered_signal, 'b', label='Ali: 1212171 \nKhalid: 1210618') 
    plt.xlabel('Time (seconds)') 
    plt.ylabel('Amplitude (mv)') 
    plt.title('Low pass Filtered ' + signal_name) 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
def cascade_filter ( signal_name: str): 
    #defning the nominator and denominator of HPF 
    H_numerator = [0] * 33 
    H_numerator[0], H_numerator[16], H_numerator[17], H_numerator[32] = -1/32, 1, -1, 1/32 
    H_denominator = [1, -1] 
    #defning the nominator and denominator of lPF 
    L_numerator = [0] * 13 
    L_numerator[0], L_numerator[6], L_numerator[12] = 1, -2, 1 
    L_denominator = [1, -2, 1] 
    # get the data array 
    amplitude, time = get_excel_data('Data_ECG_raw.xlsx', signal_name) 
    # using the output of HPF as input of LPF  
    HPF_signal = lfilter(H_numerator, H_denominator, amplitude) 
    result_signal = lfilter(L_numerator, L_denominator, HPF_signal) 
    # plot the resultant result 
    plt.figure() 

 
    plt.plot(time, result_signal, 'b', label='Ali: 1212171 \nKhalid: 1210618') 
    plt.xlabel('Time (seconds)') 
    plt.ylabel('Amplitude (mv)') 
    plt.title('High pass then low pass filtered ' + signal_name) 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
def reverse_cascade_filter ( signal_name: str): 
    #defning the nominator and denominator of HPF 
    H_numerator = [0] * 33 
    H_numerator[0], H_numerator[16], H_numerator[17], H_numerator[32] = -1/32, 1, -1, 1/32 
    H_denominator = [1, -1] 
    #defning the nominator and denominator of lPF 
    L_numerator = [0] * 13 
    L_numerator[0], L_numerator[6], L_numerator[12] = 1, -2, 1 
    L_denominator = [1, -2, 1] 
    # get the data array 
    amplitude, time = get_excel_data('Data_ECG_raw.xlsx', signal_name) 
    # using the output of LPF as input of HPF  
    LPF_signal = lfilter(L_numerator, L_denominator, amplitude) 
    result_signal = lfilter(H_numerator, H_denominator, LPF_signal) 
    # plot the resultant result 
    plt.figure() 
    plt.plot(time, result_signal, 'b', label='Ali: 1212171 \nKhalid: 1210618') 
    plt.xlabel('Time (seconds)') 
    plt.ylabel('Amplitude (mv)') 
    plt.title('Low pass then high pass filtered ' + signal_name) 
    plt.legend() 
    plt.grid(True) 
    plt.show() 
 
def Fourier_transform (signal_name: str): 
    amplitude, time = get_excel_data('Data_ECG_raw.xlsx', signal_name) 
     
    # define a number of samples to read ( to aviod non numbers read from excel file
    N=8000 
    # Compute the Fourier transform 
    fft_amplitude = np.fft.fft(amplitude[:N]) 
    fft_freq = np.fft.fftfreq(len(amplitude[:N]), d=(time[1] - time[0])) 
    # Single-sided spectrum 
    single_sided_fft_amplitude = fft_amplitude[:N // 2] 
    single_sided_fft_freq = fft_freq[:N // 2] 

 
 
    # Plot the original signal 
    plt.figure() 
    # Plot the magnitude of the single-sided Fourier transform 
    plt.plot(single_sided_fft_freq, np.abs(single_sided_fft_amplitude), 
    label='Ali: 1212171\nKhaled: 1210618') 
    plt.title(f'Fourier Transform: {signal_name}') 
    plt.xlabel('Frequency [Hz]') 
    plt.ylabel('Magnitude') 
    plt.grid() 
    plt.legend() 
     
    plt.show() 
    
 
if __name__ == '_main_': 
    Data_visualization('Data_ECG_raw.xlsx', 'ECG1') 
    #high_pass_filter('ECG2') 
    #low_pass_filter('ECG1') 
    #cascade_filter('ECG2') 
    #reverse_cascade_filter('ECG1') 
    #Fourier_transform('ECG2') 