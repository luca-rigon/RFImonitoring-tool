import glob
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from utils import *
from load_map import load_map
from read_logfile import return_frequencies_sx

# Constants:
observatory_coordinates = (11.2, 47.95, 11.9, 48.35)
clip_min = 0
clip_max = 3  

# Here, change the split number if needed(e.g too much files exist for processing)
split_num=160

def sxL_IndexToChannel(i):
    # Defined only for i==8 -> CH1 or i==9 -> CH8
    # for correct plot labeling
    if i == 8: return '1'
    else: return '8'

# Plots:
def sx_gnuplot(dataset, session, time, band_flags):
    # Generate GNU-plot of one dataset, for a given session (S/X antenna)
    # shape: variable, based on measured bands (4000 x (1+xxx))
    # channel 0: frequencies (x-axis)
    # channels 1-8: X bands, 1 & 8 are up- & low-polarized
    # channels 9-14: S bands, up-polarized

    count_voids = 0
    low_pol = 0

    plt.rcParams['lines.linestyle'] = '-'
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7  

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle(f'Single scan plot, S/X Bands: session {session}, {time[0:4]}-{time[5:7]}-{time[8:10]} {time[11:]}')

    if band_flags[-1] == 2:  # BBC15 & BBC16 allocation
        for i in range(16):
            ax = fig.add_subplot(4, 4, i+1)
            if band_flags[i]:
                if i < 8: band = 'X'
                else: 
                    band = 'S'

                label = f"{band} {i+1}u"
                ax.plot(dataset[:, 0], dataset[:, i+1-count_voids], label=label) 
                ax.set_xlim(0.01, 7.99)
                ax.set_ylim(0, 3)
                plt.xticks(range(1, 8))
                plt.text(0.85, 0.9, label, backgroundcolor='white', ha='center', va='center', transform=plt.gca().transAxes, fontsize=7)   

            else:
                count_voids += 1    
                plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

    else:                   # allocate lower bands
        for i in range(16):
            ax = fig.add_subplot(4, 4, i+1)
            if band_flags[i]:
                if i < 10: band = 'X'
                else: 
                    band = 'S'
                    low_pol = 2 
                
                if i==8 or i==9:
                    label = f"{band} "+sxL_IndexToChannel(i)+"l"    
                    ax.plot(dataset[::-1, 0], dataset[:, i+1-count_voids], label=label)             
                else: 
                    label = f"{band} {i+1-low_pol}u"
                    ax.plot(dataset[:, 0], dataset[:, i+1-count_voids], label=label) 
                ax.set_xlim(0.01, 7.99)
                ax.set_ylim(0, 3)
                plt.xticks(range(1, 8))
                plt.text(0.85, 0.9, label, backgroundcolor='white', ha='center', va='center', transform=plt.gca().transAxes, fontsize=7)   

            else:
                count_voids += 1    
                plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    plt.savefig(f'{save_session_path(session)}/{session}_gnuplot_cal_{timeformat_files(time)}.png')
    plt.close()

def sx_spectra_plot(datasets_list, session, band_flags, freq_vector, x_axis, times_label, method):
    # Plot spectrograms for a given session over all selected datasets, listed in datasets_list, for all frequencies
    # Inputs: - band_flags: list of flags representing channel allocations
    #         - freq_vector: list of frequency ranges within the band, in the required format for plotting
    #         - x_axis: x-axis label, in the required format for plotting
    #         - times_label: session duration, as label for title
    #         - method: statistical method to apply on data. Default: np.max()

    titles = ['X-band', 'S-band']
    all_channels_listed = []
    values_list = [] 
    number_channels = len([x for _, x in enumerate(band_flags) if x!=0])
    # Append each columns of the dataset to each other, excluding the x-axis
    for i in range(len(datasets_list)):
        all_channels_listed.append([])
        for j in range(1, number_channels+1):
            all_channels_listed[i].append(datasets_list[i][:,j])

    # Go through the n allocated channels, add one full array(maximum/mean/median) into another, split, get the maximum/mean/median and add it to a spare array(maximums_2)
    for j in range(len(datasets_list)):
        values_list.append([])
        for i in range(number_channels):
            k=np.array_split(all_channels_listed[j][i], split_num)
            method_func = getattr(np, method)       # Define function np.max(), np.mean() or np.median()
            b = [method_func(p) for p in k]
            values_list[j].append(b)

    # Allocate empty arrays and store them in a list
    X = [[] for _ in range(number_channels)]

    for ch in range(number_channels):
        for t in range(len(datasets_list)):
            X[ch].append(values_list[t][ch])
    X_clipped = np.clip(X, a_min=clip_min, a_max=clip_max)      # shape = (channels, times, n_split)

    # Concatenate all channels for the Y axis; X & S band, upper pol
    shift_Sband = 0         
    if band_flags[-1] != 2 and band_flags[8] == 1: shift_Sband = 2      # skip lower band allocations
    Xband_channels = []
    Sband_channels = []
    count_voids = 0
    x_count = 0
    s_count = 0

    for i in range(16):
            if band_flags[i]:
                if i < 8:
                    Xband_channels.append(X_clipped[i-count_voids,:,:].T)
                    x_count += 1
                else: 
                    if shift_Sband == 2 and i==14: break
                    Sband_channels.append(X_clipped[i-count_voids+shift_Sband,:,:].T)
                    s_count += 1
            else:
                count_voids += 1  

    Xband_concatenated = np.concatenate(Xband_channels, axis=0)
    Sband_concatenated = np.concatenate(Sband_channels, axis=0)

    band_counts = [x_count, s_count]
    all_channels = [Xband_concatenated, Sband_concatenated]
    # Normalize the data for colorbar
    norm = colors.Normalize(vmin=clip_min, vmax=clip_max)

    freq_vector = freq_vector[:x_count] + [(freq_vector[x_count][0], '')] + [('', freq_vector[x_count][1])] + freq_vector[x_count+1:]

    # generate 2 Figures: X & S     TODO fix labels
    for bnd in range(2):
        band_channels = all_channels[bnd]
        
        # Set up plotting: pcolor, labels, ticks, etc.
        fig, axs = plt.subplots(band_counts[bnd], sharex=True)
        fig.subplots_adjust(hspace=0)
        axs = axs[::-1]
        for i, ax_i in enumerate(axs):
            ci = ax_i.pcolor(band_channels[i*split_num:(i+1)*split_num,:], norm=norm, cmap='jet')
            if i < band_counts[bnd]-1:
                freq_range = freq_vector[i + bnd*(x_count+1)]
                freq_labels = [f'\n{str(freq_range[1])}\n{str(freq_range[0])}\n']
                ax_i.set_yticks([1.5])
                ax_i.set_yticklabels(freq_labels, fontsize=8)
            else:
                freq_range1 = freq_vector[i + bnd*(x_count+1)]
                freq_range2 = freq_vector[i + bnd*(x_count+1) + 1]
                freq_labels = [f'\n{freq_range1[1]}\n{freq_range1[0]}\n', f'\n{freq_range2[1]}\n{freq_range2[0]}\n']
                ax_i.set_yticks([1.5, split_num-1.5])
                ax_i.set_yticklabels(freq_labels, fontsize=8)
            ax_i.tick_params(axis='y', width=2, length=8)
            ax_i.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax_i.set_xticks(np.arange(len(datasets_list))+0.5)
            ax_i.set_xticklabels(x_axis, rotation=90, fontsize=5)

        axs[(band_counts[bnd]-1)//2].set_ylabel('MHz')  
        plt.xlabel('Timestamps')
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])    # - Create a shared colorbar      
        plt.colorbar(ci, cax=cax)
        fig.suptitle(f'Session {session}, {titles[bnd]}, upper polarization')
        plt.savefig(f'{save_session_path(session)}/{session}_spectraplot_{times_label}_{titles[bnd]}_{method}.png', bbox_inches='tight', dpi=300)
        plt.close()


# def sx_sky_plot(azimuths, elevations, datasets_list, channel_param, times_label, band_flags, freq_vector, method, clip_skyplot):
#     # Visualize skyplot of the datasets_list - files for given channel and corresponding lists of azimuth and elevation data
#     # Inputs: - channel_param: channel chosen for visualization. Can be either a specific one, or 'all' of them
#     #         - times_label: session duration, as label for title
#     #         - band_flags: list of flags representing channel allocations
#     #         - freq_vector: list of frequency ranges within the band, in the required format for plotting
#     #         - method: statistical method to apply on data. Default: np.max()
#     #         - clip_skyplot: wheter to clip the measurements onto a specific range or not

#     n_samples = len(datasets_list)
#     method_func = getattr(np, method)       # Define function np.max(), np.mean(), np.median()
#     # Compute max/mean/median for all frequencies:
#     number_channels = len([x for _, x in enumerate(band_flags) if x!=0])
#     values_channels_all = np.zeros((number_channels,n_samples))
#     for ch in range(number_channels-1):
#         for i in range(n_samples):
#             values_channels_all[ch,i] = method_func(datasets_list[i][:,ch+1])
    
#     clip_label = ''
#     if clip_skyplot == True:    
#         values_channels_all = np.clip(values_channels_all, a_min=clip_min, a_max=clip_max)
#         clip_label = '_clipped'

#     # Get the direction of the highest disturbance (i.e. get index for highest values):
#     disturbance_index = np.unravel_index(np.argmax(values_channels_all, axis=None), (number_channels,n_samples))
#     az_max = np.radians(azimuths[disturbance_index[1]])
#     el_max = elevations[disturbance_index[1]]
#     print(f' Highest disturbance {values_channels_all.max()} at ({azimuths[disturbance_index[1]]}, {elevations[disturbance_index[1]]}) - frequency channel {disturbance_index[0]+1}: {freq_vector[disturbance_index[0]%8][1]} - {freq_vector[disturbance_index[0]%8+1][0]}')

#     # define binning
#     abins = np.linspace(0,2*np.pi, 60)     # azimuth - angle; 1/6
#     rbins = np.linspace(0,90, 31)   # elevation - radius; 1/3
#     theta, R = np.meshgrid(abins, rbins)
    
#     if channel_param == 'all':
#         channel_list = list(range(1, number_channels+2))       # Analyze all channels + return highest values over all
#     else:
#         channel_list = [int(channel_param)]          # Analyze only one channel

#     for channel in channel_list:
#         # Initialize zero array and put the values at the provided Az & El indices, scaled onto the corresponding ranges (binning), for the selected channel:
        
#         if channel == 17:
#             channel_label = 'all'
#             channel_values = np.zeros(n_samples)
#             for i in range(n_samples):
#                 channel_values[i] = method_func(values_channels_all[:,i])       # Get max/mean/median over all channels for each sample
#         else:
#             channel_label = 'CH'+str(channel)
#             channel_values = values_channels_all[channel-1, :]
    
#         values = np.zeros((len(abins), len(rbins)))
#         for i in range(n_samples):
#             az_index = int(round(azimuths[i]/6)) 
#             el_index = int(round(elevations[i]/3)) 
#             if az_index == 60: az_index = 0      # close circle
#             # Only consider the highest value at each position:
#             if channel_values[i] > values[az_index, el_index]:
#                 values[az_index, el_index] = channel_values[i]

#         # Get the directions towards the highest value for the chosen channel:
#         max_index = np.unravel_index(np.argmax(values, axis=None), values.shape)
#         az_max_ch = np.radians(max_index[0]*6 + 3)
#         el_max_ch = max_index[1]*3 #+ 2

#         # Title of plot
#         if channel < 9:
#             plot_title = f'Skyplot for {session}, Band {band}, \n {freq_vector[channel-1][1]} - {freq_vector[channel][0]} MHz (H-pol)'
#         elif channel < 17:
#             plot_title = f'Skyplot for {session}, Band {band}, \n {freq_vector[channel-9][1]} - {freq_vector[channel-8][0]} MHz (V-pol)'  
#         else: 
#             plot_title = f'Skyplot for {session}, Band {band}, \n Over all frequencies ({freq_vector[0][1]} - {freq_vector[8][0]} MHz)'

#         # Create a polar plot
#         fig = plt.figure(figsize=(8, 8))
#         ax = fig.add_subplot(111, projection='polar')
#         cax = ax.contourf(theta, R, values.T, cmap='jet')
#         ax.annotate('', (az_max_ch, el_max_ch), xytext=(0,0), arrowprops=dict(facecolor='red')) # Draw arrow towards highest disturbance onto plot
#         ax.grid(False)
#         ax.set_yticklabels([]) #remove yticklabels
#         ax.set_theta_zero_location('N')
#         ax.set_theta_direction(-1)
#         ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])
#         ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
#         fig.colorbar(cax)
#         fig.suptitle(plot_title)
#         plt.savefig(f'{save_session_path(session)}/{session}_skyplot_{times_label}_{band}_{channel_label}_{method}{clip_label}.png', dpi=300)
#         plt.close()

    # load_map(observatory_coordinates, az_max, el_max)
   

def sx_remove_peaks(sessionfile, lower=True):
    # Remove calibration signals found at following given indices
    # Input - sessionfile: One single dataset-file from the session, converted into an array

    shape = sessionfile.shape   # (6400, 17)
    sessionfile_filtered = sessionfile
    # Indices for calibration bands to eliminate:
    calibration_indices = (
                        [505, 1005, 1505, 2005, 2505, 3005, 3505],       # upper channels
                        [495, 995, 1495, 1995, 2495, 2995, 3495],            # lower channels
                        [125, 625, 1125, 1625, 2125, 2625, 3125, 3625]   # all channels, if lower=False
                        )
    
    # for i in range(1, shape[1]):
    #     vec = sessionfile[:,i]
    #     res = [idx for idx, val in enumerate(vec) if val > 8.5]
    #     print(res)
    
    # Remove the peaks for the given indices:
    for k in range(1,shape[1]):
        if lower:
            if k == 9 or k == 10:   calibration_peaks = calibration_indices[1]
            else:   calibration_peaks = calibration_indices[0]
        else:
            calibration_peaks = calibration_indices[2]
        for p in calibration_peaks:
            sessionfile_filtered[p,k] = (sessionfile[p-1,k] + sessionfile[p+1,k]) / 2 # Replace peak with values close to it
    return sessionfile_filtered

def run_sx_analysis(session, doy_beginning, end_indicator, params, GNU_doy, method):
    # Make list of requested files for a given session, beginning & end time
    # Remove calibration signals when loading the dataset
    # Execute optional requests according to params - vector (GNU-plot, spectrograms, skyplot)

    # TODO: make function compatible also for sx Antennas
    files_path = get_session_path(session)
    
    # Titles for the plots
    freq_vector, band_flags = return_frequencies_sx(files_path+f'{session}wz.log')

    beginning = timeformat_files(doy_beginning)
    print(f'\n Fetching S/X Data of session ...')

    # Search for all files containing session name
    files = []
    for file in sorted(glob.glob(files_path+"*_spec.out")):
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
    interval = list(h for h in files if session[0:6]+'_wz_'+beginning+'_spec.out' <= h <= session[0:6]+'_wz_'+end+'_spec.out')
   
   # Format x-axis label s.t. it can be readable in the plot (max 50 at regular steps):
    year = str(doy_beginning)[0:4]
    labels_interval = len(interval)//50
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
        if (params[0] is True): dataset = sx_remove_peaks(f, lower=band_flags[8]) 
        else: dataset = f
        datasets_list.append(dataset)

    # GNU plots:
    if (params[1] is True):
        match = re.search(r'\d{4}\.\d{2}\.\d{2}\.\d{2}:\d{2}:\d{2}', GNU_doy)
        if match: 
            # Plot spectral frequencies at requested time:
            print(f' Plotting dataset at the time {GNU_doy}')
            GNU_time = timeformat_files(GNU_doy)
            gnu_index = files.index(f"{session}_wz_{GNU_time}_spec.out")
            sx_gnuplot(datasets_list[gnu_index], session, GNU_doy, band_flags)
        elif GNU_doy == 'all': 
            # Plot all datasets
            print(f'Plotting all datasets for the selected timespan: {year}DOY{interval[0][10:13]}, {interval[0][14:16]}:{interval[0][16:18]}:00 - {interval[-1][14:16]}:{interval[-1][16:18]}:00')
            for i, dataset in enumerate(datasets_list):
                GNU_doy_i = datetime.datetime.strptime(year + "-" + interval[i][10:13], "%Y-%j").strftime("%Y.%m.%d")
                GNU_doy_i += f'.{interval[i][14:16]}:{interval[i][16:18]}:00' 
                sx_gnuplot(dataset, session, GNU_doy_i, band_flags)
        else:
            raise ValueError('-GNUplot argument wrong. Please use format YYYY.MM.DD.hh:mm:ss, or `all`')
        
    # Plot spectrogram, if true
    if (params[2] is True):
        print(' Plotting spectrograms (X-band, S-band) of the session...')
        sx_spectra_plot(datasets_list, session, band_flags, freq_vector, x_axis, times_label, method)

    # Skyplot part:
    if (params[3] is not None):
        print(f' Plotting skyplot of the session...')
        start_id = interval[0][10:18]
        azimuths, elevations = get_summary_for_session(session, start_id, len(datasets_list))
        channel = params[3]    # choose channel here: el.[1,16]
        clip_skyplot = params[4]
        # sx_sky_plot(azimuths, elevations, datasets_list, channel, times_label, band_flags, freq_vector, method, clip_skyplot)


# run_sx_analysis('q24034', '2024.02.03.07:30:00', None, [True, True], 'all', 'max')
