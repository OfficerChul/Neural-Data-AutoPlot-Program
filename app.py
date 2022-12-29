#%% Import Libraries

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import mpld3

import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib
import math
matplotlib.rcParams['font.sans-serif'] = 'Arial'
plt.ioff()

#%% Set global variables and change current workding directory
patient_name = 'HSH2'
os.chdir(r'C:\Users\Kyochul Jang\Desktop\Purdue\Summer_2022\SNU\speechDecoding\Project')

# folder name to store result
root_path = os.getcwd()
figure_save_path = f'{root_path}/result/{patient_name}/'


# bandwidth
lowcut, highcut = 70, 150


#%% Create directories if not exists
dirs = ['result', patient_name]

for i, dir in enumerate(dirs):
    if not os.path.exists(f'{os.getcwd()}/{dir}'):
        os.mkdir(dir)
        os.chdir(f'{os.getcwd()}/{dir}')
        print(f'{i}: {dir} is created.')
    else:
        os.chdir(f'{os.getcwd()}/{dir}')
        print(f'{i}: {dir} is already existed.')
                
os.chdir(root_path)

#%% Load .mat data
from scipy.io import loadmat

eeg_data = 'HSH2_220507_synthesis_train'
file_mat = f'eeg_{eeg_data}.mat'
f = loadmat(f'export/{file_mat}')

data = f['data']
fs = f['fs'][0][0] # sampling rate: sample 2000 times in 1s # f['fs'] == [[2000]]
chs = [ch.strip() for ch in f['chs']]

#%% Load profustion events from the excel file
import pandas as pd

file_xlsx = f'events_{eeg_data}.xlsx'
df = pd.read_excel(f'export/{file_xlsx}')

#%% Print list of all markers defined by profusion
# This is how Kevin finds the onset of the production session
marker_id = 10
idx_first_marker = 30 # manually found from excel by Kevin -> the time when the experiement starts(?)
shift_seconds = 4 # number of second from study start time # I don't know why Kevin subtracts 4s from the original onset

marker_df = df[df['EventTypeID'] == marker_id]

# marker_df['StartSecond'].loc[30] -> session start time
# shift_seconds -> the time recording start after press recording button
t0 = marker_df['StartSecond'].loc[30] - shift_seconds

#%% Trim data from start time to end time of production session. Data only includes the first 10 words.
start_sample = np.where(data[-1] > t0)[0][0] # data[-1] contains the timestamp of the experiment

words_num = 10
tn = t0 + 4 * words_num
end_sample = np.where(data[-1] <= tn)[0][-1] # total fs in 40s == 1311958 --- 1

# The data contains artifacts at the beginning of the first word.
# so move one word forward from the start point
start_sample_of_second_word = start_sample + (fs * 4)

duration_seconds = 4 * (words_num) # 10 words
duration_samples = int(fs*duration_seconds)
end_sample = start_sample + duration_samples # Add fs of start point and the duration --- 2

prod_session = data[:, start_sample_of_second_word:end_sample] # Extract the date of 40s --- used 2
t_plot = np.arange(4, duration_seconds, 1/fs) # Every element is 1/fs: 

#%% DEMO ANALYSIS STARTS HERE
# Define useful signal processing functions
from scipy.signal import butter, filtfilt, iirnotch, sosfilt, stft
import time

def notch(y, f0=60, Q=30, fs=2000):
    b, a = iirnotch(f0, Q, fs)
    y_notched = filtfilt(b, a, y)
    return y_notched

def bandpass(y, lowcut, highcut, fs=2000, order=5):
    nyq = 0.5 * fs # can observe the half of the original sampling rate because of the nyquist theorem
    low = lowcut / nyq
    high = highcut / nyq
    Wn = [low, high]
    sos = butter(N=order, Wn=Wn, analog=False, output='sos', btype='band')
    y_filt = sosfilt(sos, y)
    return y_filt

def compute_stft(y, fs=2000, step_seconds=0.016, window_seconds=0.02, step_hertz=15.625):
    """
    Compute STFT of audio waveform with desired resolution
    """
    tic = time.time()
    overlap_seconds = window_seconds - step_seconds
    step = int(step_seconds * fs) # useful in librosa and spsi
    nperseg = int(window_seconds * fs) # useful in scipy.signal.stft
    noverlap = int(overlap_seconds * fs) # useful in scipy.signal.stft
    nfft = int(fs/step_hertz) # Optimal if power of 2: 1024 = 16000/15.625
    freqs, times, S = stft(y, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    toc = time.time()
    print(f'Audio spectrogram has shape {S.shape} | Elapsed time = {round(toc-tic, 3)} seconds')
    return freqs, times, S, step, nperseg, noverlap, nfft

def make_dict(chs):
    label = 'label'
    value = 'value'
    result = []
    for i in chs:
        if i.isdigit():
            temp = {}
            temp[label] = i
            temp[value] = i
            result.append(temp)
        else:
            print(i)
        
    return result
    
#%%
init_fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.grid()
init_item = mpld3.fig_to_html(init_fig)

channels_dropdown = make_dict(chs)

radio_options = [{'label': 'Raw', 'value': 'Raw'},
       {'label': 'PSD', 'value': 'PSD'},
       {'label': 'Spectrogram', 'value': 'Spectrogram'},]

#%% 

app = dash.Dash()
logo_path = 'assets/logo.jpg'
electrode_location = 'assets/HSH2 Placement Neurologist.png'
snu_logo_location = 'assets/snu-logo.png'

app.layout = html.Div(
    style={'backgroundColor': '#E3E3E3'},
    className='container',
    children=[
            
            html.Div(children=[
                html.Div(className = 'header', children=[
                    html.Img(src=logo_path, className='logo'),
                    html.H1(children = f'Neural Data Dashboard',
                            className='title',
                            style = {
                                'textAlign': 'center',
                                'color': '#93ABB9'}
                            ),
                    html.Img(
                            src=snu_logo_location, 
                            
                             ),
                    html.H4(className='cr', children = 'KCJ © 2022'),
                    ]),
            
                
                html.H3('Please Choose a Channel'),
                html.Div(className='option', 
                     children=[
                        dcc.Dropdown(
                            id = 'channel_select',
                            options = channels_dropdown,
                            placeholder = 'Please Choose a Channel...',
                            value=None,
                            multi=True,
                            style={
                                    'width': '300px'
                                }
                            ),
                        dcc.RadioItems(
                            radio_options, 
                            inline=True,
                            id = 'plot_method',
                            value='Raw',
                            ),
                    ]),
                
                html.Div(className='pictures', children = [
                    html.Iframe(
                            id='neural_data_plot',
                            srcDoc=init_item,
                            style={'height': '80vh'},
                        ),
                    html.Img(
                            src=electrode_location, 
                            style={'width': '50%', 'height': '80vh'},
                             ),
                    
                    ])
                
                ])
            
])

@app.callback(
    Output('neural_data_plot', 'srcDoc'),
    Input(component_id='channel_select', component_property='value'),
    Input(component_id='plot_method', component_property='value')
)

def plot_neural_data(channels, plot_method):
    print(f'channels: {channels[0]}')
    print(f'hihi: {len(channels)}')
    nrows = ncols = math.ceil(math.sqrt(len(channels)))
    # ch = chs.index(channels)
    
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.5, 6.5))
    chn_count = 0  
    
    
    if len(channels) == 1:
        if plot_method == 'Raw':
            ax.plot(t_plot, prod_session[chs.index(channels[chn_count])])
        elif plot_method == 'PSD':
            win_seconds = 0.1
            win_samples = int(fs*win_seconds)
            bp_data = bandpass(notch(prod_session[chs.index(channels[chn_count])]), lowcut, highcut)
            psd_data = pd.Series(bp_data).rolling(window=win_samples).var()
            ax.plot(t_plot, psd_data)
        elif plot_method == 'Spectrogram':
            freqs, times, S, step, nperseg, noverlap, nfft = compute_stft(prod_session[chs.index(channels[chn_count])])
            Sxx = abs(S) # Sxx: magnitude spectrogram / S: stft with complex values
            
            fmax = 200
            idx_fmax = np.where(freqs>fmax)[0][0]
            baseline = np.mean(Sxx[:idx_fmax,:fs], axis=1)
            Sxx_change = Sxx[:idx_fmax]/baseline[:,None]
            tmin = 4
            tmax = 39.984
            times = times + 4
            print(f'tmin: {np.where(times>tmin)}')
            print(f'tmax: {np.where(times>tmax)}')
            print(f'times: {times}')
            idx_tmin = np.where(times>tmin)[0][0]
            idx_tmax = np.where(times>tmax)[0][0]
            freqs_plot = freqs[:idx_fmax]
            times_plot = times[idx_tmin:idx_tmax]
            Sxx_plot = Sxx_change[:,idx_tmin:idx_tmax]
            
            ax.pcolormesh(times_plot, freqs_plot, Sxx_plot, cmap='magma', shading='auto')
        ax.grid()
        ax.set_title(f'{channels[chn_count]}th ch')
        ax.set_xlim([4, t_plot[-1]])
        # ax[i][j].set_ylim([])
        # ax.set_xticks([])
        # ax.set_yticks([])

        # chn_count += 1
    else:
        for i in range(nrows):
            for j in range(ncols):
                if chn_count >= len(channels):
                    print(f'i: {i}, j: {j}')
                    break;
                print(f'i: {i}, j: {j}, nols: {ncols}, nrows: {nrows}')
                if plot_method == 'Raw':
                    ax[i][j].plot(t_plot, prod_session[chs.index(channels[chn_count])])
                elif plot_method == 'PSD':
                    win_seconds = 0.1
                    win_samples = int(fs*win_seconds)
                    bp_data = bandpass(notch(prod_session[chs.index(channels[chn_count])]), lowcut, highcut)
                    psd_data = pd.Series(bp_data).rolling(window=win_samples).var()
                    ax[i][j].plot(t_plot, psd_data)
                elif plot_method == 'Spectrogram':
                    freqs, times, S, step, nperseg, noverlap, nfft = compute_stft(prod_session[chs.index(channels[chn_count])])
                    Sxx = abs(S) # Sxx: magnitude spectrogram / S: stft with complex values
                    
                    fmax = 200
                    idx_fmax = np.where(freqs>fmax)[0][0]
                    baseline = np.mean(Sxx[:idx_fmax,:fs], axis=1)
                    Sxx_change = Sxx[:idx_fmax]/baseline[:,None]
                    tmin = 4
                    tmax = 39.984
                    times = times + 4
                    print(f'tmin: {np.where(times>tmin)}')
                    print(f'tmax: {np.where(times>tmax)}')
                    print(f'times: {times}')
                    idx_tmin = np.where(times>tmin)[0][0]
                    idx_tmax = np.where(times>tmax)[0][0]
                    freqs_plot = freqs[:idx_fmax]
                    times_plot = times[idx_tmin:idx_tmax]
                    Sxx_plot = Sxx_change[:,idx_tmin:idx_tmax]
                    
                    ax[i][j].pcolormesh(times_plot, freqs_plot, Sxx_plot, cmap='magma', shading='auto')
                
                ax[i][j].grid()
                ax[i][j].set_title(f'{channels[chn_count]}th ch')
                ax[i][j].set_xlim([4, t_plot[-1]])
                
                chn_count += 1
        

    plt.tight_layout()
    # plt.ion()
    # plt.show()
    
    return_item = mpld3.fig_to_html(fig)
    
    return return_item




if __name__ == '__main__':
    app.run_server()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    