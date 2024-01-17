import glob
import os
import re
import argparse
import time
import datetime
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from utils import *
from load_map import load_map

# Constants:
observatory_coordinates = (11.2, 47.95, 11.9, 48.35)
clip_min = 0
clip_max = 3  

# Plots:
def gnuplot(dataset, session, time, band):
    # Generate GNU-plot of one dataset, for a given session, time & spectral band
    # shape: (6400 x (1+16))
    # channel 0: frequencies (x-axis)
    # channels 1-8: Horizontal polarization
    # channels 9-17: Vertical polarization

    plt.rcParams['lines.linestyle'] = '-'
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle(f'Single scan plot: session {session}, {time[0:4]}-{time[5:7]}-{time[8:10]} {time[11:]} - Band {band}')
    for i in range(16):
        if i < 8:
            label = "H-pol"
        else:
            label = "V-pol"
            
        ax = fig.add_subplot(4, 4, i+1)
        ax.plot(dataset[:, 0], dataset[:, i+1], label=label)
        ax.set_xlim(-1, 33)
        ax.set_ylim(0, 3)
        plt.xticks(range(0, 34, 5))
        plt.text(0.85, 0.9, label, ha='center', va='center', transform=plt.gca().transAxes, fontsize=7)            

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    plt.savefig(f'{save_session_path(session)}/{session}_gnuplot_{timeformat_files(time)}_{band}.png')
    plt.close()


def spectra_plot(datasets_list, band, freq_vector, x_axis, times_label, method):
    # Plot spectrograms for a given session over all selected datasets, listed in datasets_list, for all frequencies of the chosen band
    # Inputs: - freq_vector: list of frequency ranges within the band, in the required format for plotting
    #         - x_axis: x-axis label, in the required format for plotting
    #         - times_label: session duration, as label for title
    #         - method: statistical method to apply on data. Default: np.max()

    titles = ['Horizontal', 'Vertical']
    all_channels_listed = []
    values_list = [] 
    # Append each columns of the dataset to each other, excluding the x-axis
    for i in range(len(datasets_list)):
        all_channels_listed.append([])
        for j in range(1, 17):
            all_channels_listed[i].append(datasets_list[i][:,j])

    # Here, change the split number if needed(e.g too much files exist for processing)
    split_num=160
    # Go through the 16 channels, add one full array(maximum/mean/median) into another, split, get the maximum/mean/median and add it to a spare array(maximums_2)
    for j in range(len(datasets_list)):
        values_list.append([])
        for i in range(16):
            k=np.array_split(all_channels_listed[j][i], split_num)
            method_func = getattr(np, method)       # Define function np.max(), np.mean() or np.median()
            b = [method_func(p) for p in k]
            values_list[j].append(b)

    X1=[]
    X2=[]
    X3=[]
    X4=[]
    X5=[]
    X6=[]
    X7=[]
    X8=[]
    X9=[]
    X10=[]
    X11=[]
    X12=[]
    X13=[]
    X14=[]
    X15=[]
    X16=[]
    for t in range(len(datasets_list)):
        X1.append(values_list[t][0])
        X2.append(values_list[t][1])
        X3.append(values_list[t][2])
        X4.append(values_list[t][3])
        X5.append(values_list[t][4])
        X6.append(values_list[t][5])
        X7.append(values_list[t][6])
        X8.append(values_list[t][7])
        X9.append(values_list[t][8])
        X10.append(values_list[t][9])
        X11.append(values_list[t][10])
        X12.append(values_list[t][11])
        X13.append(values_list[t][12])
        X14.append(values_list[t][13])
        X15.append(values_list[t][14])
        X16.append(values_list[t][15])
            
    # limit the values between clip_min and clip_max (0,3)  
    X1_clipped = np.clip(X1, a_min=clip_min, a_max=clip_max).T
    X2_clipped = np.clip(X2, a_min=clip_min, a_max=clip_max).T
    X3_clipped = np.clip(X3, a_min=clip_min, a_max=clip_max).T
    X4_clipped = np.clip(X4, a_min=clip_min, a_max=clip_max).T
    X5_clipped = np.clip(X5, a_min=clip_min, a_max=clip_max).T
    X6_clipped = np.clip(X6, a_min=clip_min, a_max=clip_max).T
    X7_clipped = np.clip(X7, a_min=clip_min, a_max=clip_max).T
    X8_clipped = np.clip(X8, a_min=clip_min, a_max=clip_max).T
    X9_clipped = np.clip(X9, a_min=clip_min, a_max=clip_max).T
    X10_clipped = np.clip(X10, a_min=clip_min, a_max=clip_max).T
    X11_clipped = np.clip(X11, a_min=clip_min, a_max=clip_max).T
    X12_clipped = np.clip(X12, a_min=clip_min, a_max=clip_max).T
    X13_clipped = np.clip(X13, a_min=clip_min, a_max=clip_max).T
    X14_clipped = np.clip(X14, a_min=clip_min, a_max=clip_max).T
    X15_clipped = np.clip(X15, a_min=clip_min, a_max=clip_max).T
    X16_clipped = np.clip(X16, a_min=clip_min, a_max=clip_max).T

    # Concatenate all 8 channels for the Y axis; horizontal & vertical polarization
    hor_pol_channels = np.concatenate((X1_clipped, X2_clipped, X3_clipped, X4_clipped, X5_clipped, X6_clipped, X7_clipped, X8_clipped))
    vert_pol_channels = np.concatenate((X9_clipped, X10_clipped, X11_clipped, X12_clipped, X13_clipped, X14_clipped, X15_clipped, X16_clipped))
    all_channels = [hor_pol_channels, vert_pol_channels]
    # Normalize the data for colorbar
    norm = colors.Normalize(vmin=clip_min, vmax=clip_max)

    # generate 2 Figures: 1 for H- and 1 for V-polarization
    for pol in range(2):
        polarization_channels = all_channels[pol]

        # Set up plotting: pcolor, labels, ticks, etc.
        fig, axs = plt.subplots(8, sharex=True)
        fig.subplots_adjust(hspace=0)
        axs = axs[::-1]
        for i, ax_i in enumerate(axs):
            ci = ax_i.pcolor(polarization_channels[i*split_num:(i+1)*split_num], norm=norm, cmap='jet')
            if i < 7:
                freq_range = freq_vector[i]
                freq_labels = [f'\n{str(freq_range[1])}\n{str(freq_range[0])}\n']
                ax_i.set_yticks([1.5])
                ax_i.set_yticklabels(freq_labels, fontsize=8)
            else:
                freq_range1 = freq_vector[i]
                freq_range2 = freq_vector[i+1]
                freq_labels = [f'\n{freq_range1[1]}\n{freq_range1[0]}\n', f'\n{freq_range2[1]}\n{freq_range2[0]}\n']
                ax_i.set_yticks([1.5, split_num-1.5])
                ax_i.set_yticklabels(freq_labels, fontsize=8)
            ax_i.tick_params(axis='y', width=2, length=8)
            ax_i.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax_i.set_xticks(np.arange(len(datasets_list))+0.5)
            ax_i.set_xticklabels(x_axis, rotation=90, fontsize=5)

        axs[3].set_ylabel('MHz')  
        plt.xlabel('File names')
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])    # - Create a shared colorbar      
        plt.colorbar(ci, cax=cax)
        fig.suptitle(f'Session {session}, {titles[pol]} Polarization, Band {band}')
        plt.savefig(f'{save_session_path(session)}/{session}_spectraplot_{times_label}_{band}_{titles[pol]}_{method}.png', dpi=300)
        # plt.show()


def sky_plot(azimuths, elevations, datasets_list, band, channel_param, times_label, freq_vector, method, clip_skyplot):
    # Visualize skyplot of the datasets_list - files for given channel and corresponding lists of azimuth and elevation data
    # Inputs: - channel_param: channel chosen for visualization. Can be either a specific one, or 'all' of them
    #         - times_label: session duration, as label for title
    #         - freq_vector: list of frequency ranges within the band, in the required format for plotting
    #         - method: statistical method to apply on data. Default: np.max()
    #         - clip_skyplot: wheter to clip the measurements onto a specific range or not

    n_samples = len(datasets_list)
    method_func = getattr(np, method)       # Define function np.max(), np.mean(), np.median()
    # Compute max/mean/median for all frequencies:
    values_channels_all = np.zeros((16,n_samples))
    for ch in range(15):
        for i in range(n_samples):
            values_channels_all[ch,i] = method_func(datasets_list[i][:,ch+1])
    
    clip_label = ''
    if clip_skyplot == True:    
        values_channels_all = np.clip(values_channels_all, a_min=clip_min, a_max=clip_max)
        clip_label = '_clipped'

    # Get the direction of the highest disturbance (i.e. get index for highest values):
    disturbance_index = np.unravel_index(np.argmax(values_channels_all, axis=None), (16,n_samples))
    az_max = np.radians(azimuths[disturbance_index[1]])
    el_max = elevations[disturbance_index[1]]
    print(f' Highest disturbance {values_channels_all.max()} at ({azimuths[disturbance_index[1]]}, {elevations[disturbance_index[1]]}) - frequency channel {disturbance_index[0]+1}: {freq_vector[disturbance_index[0]%8][1]} - {freq_vector[disturbance_index[0]%8+1][0]}')

    # define binning
    abins = np.linspace(0,2*np.pi, 60)     # azimuth - angle; 1/6
    rbins = np.linspace(0,90, 31)   # elevation - radius; 1/3
    theta, R = np.meshgrid(abins, rbins)
    
    if channel_param == 'all':
        channel_list = list(range(1, 18))       # Analyze all channels + return highest values over all
    else:
        channel_list = [int(channel_param)]          # Analyze only one channel

    for channel in channel_list:
        # Initialize zero array and put the values at the provided Az & El indices, scaled onto the corresponding ranges (binning), for the selected channel:
        
        if channel == 17:
            channel_label = 'all'
            channel_values = np.zeros(n_samples)
            for i in range(n_samples):
                channel_values[i] = method_func(values_channels_all[:,i])       # Get max/mean/median over all channels for each sample
        else:
            channel_label = 'CH'+str(channel)
            channel_values = values_channels_all[channel-1, :]
    
        values = np.zeros((len(abins), len(rbins)))
        for i in range(n_samples):
            az_index = int(round(azimuths[i]/6)) 
            el_index = int(round(elevations[i]/3)) 
            if az_index == 60: az_index = 0      # close circle
            # Only consider the highest value at each position:
            if channel_values[i] > values[az_index, el_index]:
                values[az_index, el_index] = channel_values[i]

        # Get the directions towards the highest value for the chosen channel:
        max_index = np.unravel_index(np.argmax(values, axis=None), values.shape)
        az_max_ch = np.radians(max_index[0]*6 + 3)
        el_max_ch = max_index[1]*3 #+ 2

        # Title of plot
        if channel < 9:
            plot_title = f'Skyplot for {session}, Band {band}, \n {freq_vector[channel-1][1]} - {freq_vector[channel][0]} MHz (H-pol)'
        elif channel < 17:
            plot_title = f'Skyplot for {session}, Band {band}, \n {freq_vector[channel-9][1]} - {freq_vector[channel-8][0]} MHz (V-pol)'  
        else: 
            plot_title = f'Skyplot for {session}, Band {band}, \n Over all frequencies ({freq_vector[0][1]} - {freq_vector[8][0]} MHz)'

        # Create a polar plot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='polar')
        cax = ax.contourf(theta, R, values.T, cmap='jet')
        ax.annotate('', (az_max_ch, el_max_ch), xytext=(0,0), arrowprops=dict(facecolor='red')) # Draw arrow towards highest disturbance onto plot
        ax.grid(False)
        ax.set_yticklabels([]) #remove yticklabels
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        fig.colorbar(cax)
        fig.suptitle(plot_title)
        plt.savefig(f'{save_session_path(session)}/{session}_skyplot_{times_label}_{band}_{channel_label}_{method}{clip_label}.png', dpi=300)
        plt.close()

    # load_map(observatory_coordinates, az_max, el_max)
   

def run_analysis(session, doy_beginning, end_indicator, spec, params, GNU_doy, method):
    # Make list of requested files for a given session, beginning & end time, as well as spectral band
    # Remove calibration signals when loading the dataset
    # Execute optional requests according to params - vector (GNU-plot, spectrograms, skyplot)

    # Titles for the plots
    band, freq_vector = return_band(spec)

    # Prepare lists of datasets
    beginning = timeformat_files(doy_beginning)
    print(f'\n Fetching Data of session, Band {band}...')

    # Create the output file list from the folder
    files = []
    files_path = get_session_path(session)

    # Search for all files containing session name
    for file in sorted(glob.glob(files_path+"*scansection_"+spec+".spec")):
        file_index = file.find("\\")
        file_name = file[file_index + 1:]
        files.append(file_name)

    # Check end indicator 
    if end_indicator == None:     
        doy_end = files[-1]
        end = doy_end[10:18]
    else: 
        end = end_indicator
    
    times_label = beginning[4:] + '-' + end[4:]
    
    # Shrink the list according to the user input
    interval = list(h for h in files if session[0:6]+'_ws_'+beginning+'_scansection_' +
                    spec+'.spec' <= h <= session[0:6]+'_ws_'+end+'_scansection_'+spec+'.spec')
   
   # Format x-axis label s.t. it can be readable in the plot (max 50 at regular steps):
    year = str(doy_beginning)[0:4]
    labels_interval = int(len(interval)/50)
    if labels_interval == 0: labels_interval = 1    # <- less than 50 labels; visualize them all
    
    x_axis = []
    for i,x in enumerate(interval):
        if i%labels_interval == 0:
            x_axis.append(datetime.datetime(int(year), 1, 1) + datetime.timedelta(int(x[10:13]) - 1, minutes=int(x[16:18]), hours=int(x[14:16])))
        else:
            x_axis.append('')

    # Read the spec values from the single datasets, make a list out of them
    # Remove calibration signals if param[0] is True:
    datasets_list = []
    for i in range(len(interval)):
        f = np.loadtxt(files_path+interval[i])
        if (params[0] is True): dataset = remove_peaks(f)
        else: dataset = f
        datasets_list.append(dataset)

    # GNU plots:
    if (params[1] is True):
        match = re.search(r'\d{4}\.\d{2}\.\d{2}\.\d{2}:\d{2}:\d{2}', GNU_doy)
        if match: 
            # Plot spectral frequencies at requested time:
            print(f' Plotting dataset at the time {GNU_doy}')
            GNU_time = timeformat_files(GNU_doy)
            gnu_index = files.index(f"{session}_ws_{GNU_time}_scansection_{spec}.spec")
            gnuplot(datasets_list[gnu_index], session, GNU_doy, band)
        elif GNU_doy == 'all':
            # Plot all datasets
            print(f'Plotting all datasets for the selected timespan')
            for i, dataset in enumerate(datasets_list):
                GNU_doy_i = datetime.datetime.strptime(year + "-" + interval[i][10:13], "%Y-%j").strftime("%Y.%m.%d")
                GNU_doy_i += f'.{interval[i][14:16]}:{interval[i][16:18]}:00' 
                print(GNU_doy_i)
                gnuplot(dataset, session, GNU_doy_i, band)
        else:
            raise ValueError('-GNUplot argument wrong. Please use format YYYY.MM.DD.hh:mm:ss, or `all`')
        
    # Plot spectrogram, if true
    if (params[2] is True):
        print(' Plotting spectrograms (pol H and V) of the session...')
        spectra_plot(datasets_list, band, freq_vector, x_axis, times_label, method)

    # Skyplot part:
    if (params[3] is not None):
        print(f' Plotting skyplot of the session...')
        start_id = interval[0][10:18]
        azimuths, elevations = get_summary_for_session(session, start_id, len(datasets_list))
        channel = params[3]    # choose channel here: el.[1,16]
        clip_skyplot = params[4]
        sky_plot(azimuths, elevations, datasets_list, band, channel, times_label, freq_vector, method, clip_skyplot)
        

if __name__ =='__main__':

    bands = ['0', '1', '2', '3']    # A, B, C, D
    GNUplot = False
    spectraplot = False
    skyplot = None
    filter_calibration = True
    method = 'max'
    clip_skyplot = False

    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    # Session informations:
    ap.add_argument("-s", "--foperand", required=True,
                    help="Start time for the session")
    ap.add_argument("-e", "--soperand", required=False,
                    help="End time for the session")
    ap.add_argument("-session", "--toperand", required=True,
                    help="Session information")
    ap.add_argument("-d", "--doperand", required=False,
                    help="Duration of session")
    

    # Optional argument: Specify if the calibration signals are to be filtered out. Default=True
    ap.add_argument("-removeCal", "--caliboperand", required=False,
                    help="Visualize spectral data for one session file, for given input times")
    
    # Optional argument: Specify the method for evaluating signal outliers; can be 'max', 'mean', 'median'. Default='max'
    ap.add_argument("-method", "--methodoperand", required=False,
                    help="Analysis method")

    # Optional argument: Visualize spectral data for a single session time, or all of them. Input has to be datetime-format/'all'
    ap.add_argument("-GNUplot", "--gnuoperand", required=False,
                    help="Visualize spectral data for one session file, for given input times")
    
    # Optional argument: Visualize spectrogram of session. Only valid/true for input = 1-16, requesting the corresponding channel
    ap.add_argument("-spectrogram", "--spectroperand", required=False,
                    help="activate spectrogram-plots")
    
    # Optional argument: Create skyplot for session. Only valid/true for input = 1
    ap.add_argument("-skyplot", "--skyoperand", required=False,
                    help="activate skyplot")

    # Optional argument: Clip skyplot data. Only valid/true for input = 1, default: False
    ap.add_argument("-sky_clip", "--clipoperand", required=False,
                    help="activate skyplot")
    args = vars(ap.parse_args())
    

    session = args['toperand']
    doy_beginning = args['foperand']    

    # Check the end indicator. Can be either the duration (-d) option or end option(-e).
    if(args['doperand'] is not None):
        doy_end=args['foperand']
        format = "%Y.%m.%d.%H:%M:%S"
        dt_e = datetime.datetime.strptime(doy_end, format)
        tt_e = (dt_e+timedelta(hours=int(args['doperand'].partition("h")[0]))).timetuple()
        # Control the given hour format to prevent type errors
        hours = ''
        mins = ''
        # Hours format:
        if tt_e.tm_hour<10:
            hours=str(0)+str(tt_e.tm_hour)
        else:
            hours=str(tt_e.tm_hour)
        # Minutes format:
        if tt_e.tm_min<10:
            mins=str(0)+str(tt_e.tm_min)
        else:
            mins=str(tt_e.tm_min)
        end_indicator = str(tt_e.tm_yday)+'-'+hours+mins
    elif(args['soperand'] is not None):  
        doy_end = args['soperand']
        end_indicator = timeformat_files(doy_end)
    else: end_indicator = None                       

    # Check optional arguments:
    GNU_doy = None
    if(args['gnuoperand'] is not None):
        GNUplot = True
        GNU_doy = args['gnuoperand']
    if(args['spectroperand'] == '1'):
        spectraplot = True
    if(args['skyoperand'] is not None):
        skyplot = args['skyoperand']

    if(args['caliboperand'] == '0'):
        filter_calibration = False

    if(args['methodoperand'] is not None):
        method = args['methodoperand']

    if(args['clipoperand'] == '1'):
        clip_skyplot = True
        print('clip on')
    
    params = [filter_calibration, GNUplot, spectraplot, skyplot, clip_skyplot]

    start_time = time.time()

    print(f'\nAnalysis for session {session}, start time: {doy_beginning}')
    for i in range(len(bands)):
       run_analysis(session, doy_beginning, end_indicator, bands[i], params, GNU_doy, method)
    end_time = time.time()

    print(f'\nAnalysis Done! Total duration: {round(end_time - start_time, 3)} s \n')