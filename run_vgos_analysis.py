import glob
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from utils import *

# Constants:
observatory_location = 'Wettzell, Bad Kotzting'
# Here, change the split number if needed(e.g too much files exist for processing)
split_num=160

# Plots:
def vgos_gnuplot(dataset, session, time, band, figsave=True):
    # Generate GNU-plot of one dataset, for a given session, time & spectral band (VGOS)
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
        if i < 8: label = "H-pol"
        else:     label = "V-pol"
            
        ax = fig.add_subplot(4, 4, i+1)
        ax.plot(dataset[:, 0], dataset[:, i+1], label=label)
        ax.set_xlim(-1, 33)
        ax.set_ylim(0, 3)
        plt.xticks(range(0, 34, 5))
        plt.text(0.85, 0.9, label, ha='center', va='center', transform=plt.gca().transAxes, fontsize=7)            

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.2, hspace=0.2)
    if figsave:
        plt.savefig(f'{save_session_path(session)}/{session}_gnuplot_{timeformat_files(time)}_{band}.png')
        plt.close()
    else: 
        plt.show()


def vgos_spectra_plot(datasets_list, session, band, freq_vector, x_axis, times_label, method, figsave=True):
    # Plot spectrograms for a given session over all selected datasets, listed in datasets_list, for all frequencies of the chosen band
    # Inputs: - freq_vector: list of frequency ranges within the band, in the required format for plotting
    #         - x_axis: x-axis label, in the required format for plotting
    #         - times_label: session duration, as label for title
    #         - method: statistical method to apply on data. Default: np.max()

    titles = ['Horizontal', 'Vertical']
    all_channels_listed = []
    values_list = [] 

    clip_min = 0
    clip_max = 3  

    # Append each columns of the dataset to each other, excluding the x-axis
    for i in range(len(datasets_list)):
        all_channels_listed.append([])
        for j in range(1, 17):
            all_channels_listed[i].append(datasets_list[i][:,j])

    # Go through the 16 channels, add one full array(maximum/mean/median) into another, split, get the maximum/mean/median and add it to a spare array(maximums_2)
    for j in range(len(datasets_list)):
        values_list.append([])
        for i in range(16):
            k=np.array_split(all_channels_listed[j][i], split_num)
            method_func = getattr(np, method)       # Define function np.max(), np.mean() or np.median()
            b = [method_func(p) for p in k]
            values_list[j].append(b)

    # Allocate empty arrays and store them in a list
    X = [[] for _ in range(16)]

    for ch in range(16):
        for t in range(len(datasets_list)):
            X[ch].append(values_list[t][ch])
    X_clipped = np.clip(X, a_min=clip_min, a_max=clip_max)      # shape = (channels, times, n_split)
    
    # Concatenate all 8 channels for the Y axis; horizontal & vertical polarization
    X_hor = []
    X_vert = []

    for i in range(16):
            if i < 8:
                X_hor.append(X_clipped[i,:,:].T)
            else: 
                X_vert.append(X_clipped[i,:,:].T)

    hor_pol_channels = np.concatenate(X_hor, axis=0)
    vert_pol_channels = np.concatenate(X_vert, axis=0)
    all_channels = [hor_pol_channels, vert_pol_channels]

    norm = colors.Normalize(vmin=clip_min, vmax=clip_max) # Normalize the data for colorbar

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
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])    # Create a shared colorbar      
        plt.colorbar(ci, cax=cax)
        fig.suptitle(f'Session {session}, {titles[pol]} Polarization, Band {band}')
        if figsave:
            plt.savefig(f'{save_session_path(session)}/{session}_spectraplot_{times_label}_{band}_{titles[pol]}_{method}.png',  bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()

def vgos_sky_plot(azimuths, elevations, datasets_list, session, band, channel_param, times_label, freq_vector, method, clip_range, obs_map=False, figsave=True):
    # Visualize skyplot of the datasets_list - files for given channels and corresponding list of azimuth and elevation data
    # Inputs: - channel_param: skyplot type chosen for visualization. Can be either 'per_channel', or 'all' for all taken together
    #         - times_label: session duration, as label for title
    #         - freq_vector: list of frequency ranges within the band, in the required format for plotting
    #         - method: statistical method to apply on data. Default: np.max()
    #         - clip_skyplot: wheter to clip the measurements onto a specific range or not
    #         - obs_map: additional parameter, if the main disturbances are to be shown on the map for a given location (Set parameter on top)

    number_channels = 16
    n_samples = len(datasets_list)
    method_func = getattr(np, method)       # Define function np.max(), np.mean(), np.median()

    # Clipping settings:
    clipping = False
    clip_min, clip_max = [0,1000]
    if (clip_range is not None):
        clip_min, clip_max = clip_range    
        clipping = True

    # Compute max/mean/median for all frequencies:
    values_channels_all = np.zeros((16,n_samples))
    for ch in range(number_channels):
        for i in range(n_samples):
            values_channels_all[ch,i] = method_func(datasets_list[i][:,ch+1])

    # Get the directions of the 5 highest disturbances (i.e. get indices for highest values):
    n_dist = 5
    all_vals_copy = values_channels_all.copy()
    dist_indices = []
    az_max_list = []
    el_max_list = []

    for _ in range(n_dist):
        disturbance_index = np.unravel_index(np.argmax(all_vals_copy, axis=None), (number_channels,n_samples))
        dist_indices.append(disturbance_index)
        az_max_list.append(np.radians(azimuths[disturbance_index[1]]))
        el_max_list.append(elevations[disturbance_index[1]])
        all_vals_copy[disturbance_index[0], disturbance_index[1]] = 0
    print(f' Highest disturbance {values_channels_all.max()} at ({azimuths[dist_indices[0][1]]}, {el_max_list[0]}) - frequency channel {dist_indices[0][0]+1}: {freq_vector[dist_indices[0][0]%8][1]} - {freq_vector[dist_indices[0][0]%8+1][0]}')

    # define binning
    abins = np.linspace(0,2*np.pi, 60)     # azimuth - angle; 1/6
    rbins = np.linspace(0,90, 31)          # elevation - radius; 1/3
    theta, R = np.meshgrid(abins, rbins)
    
    if 'all' in channel_param:
        # Run analysis over all channels
        channel_values = np.zeros(n_samples)
        for i in range(n_samples):
            channel_values[i] = method_func(values_channels_all[:,i])       # Get max/mean/median over all channels for each sample
        values, az_max, el_max = return_polar_values(azimuths, elevations, abins, rbins, channel_values)
        if clipping: values = np.clip(values, a_min=clip_min, a_max=clip_max)

        plot_title = f'Skyplot for {session}, Band {band}, \n Over all frequencies ({freq_vector[0][1]} - {freq_vector[8][0]} MHz)'
        polar_plot(session, theta, R, values, az_max, el_max, plot_title, times_label, '_'+band, '_all', method, clip_range, figsave=figsave)


    if 'per_channel' in channel_param:
        # Run analysis for every single channel separately
        channel_list = np.arange(1, number_channels+1)

        for channel in channel_list:
            plot_title = ''          
            channel_label = '_CH'+str(channel)
            
            channel_values = values_channels_all[channel-1, :]
            values, az_max_ch, el_max_ch = return_polar_values(azimuths, elevations, abins, rbins, channel_values)
            if clipping: values = np.clip(values, a_min=clip_min, a_max=clip_max)

            # Title of plot
            if channel < 9:
                plot_title = f'Skyplot for {session}, Band {band}, \n {freq_vector[channel-1][1]} - {freq_vector[channel][0]} MHz (H-pol)'
            else:
                plot_title = f'Skyplot for {session}, Band {band}, \n {freq_vector[channel-9][1]} - {freq_vector[channel-8][0]} MHz (V-pol)'  
            polar_plot(session, theta, R, values, az_max_ch, el_max_ch, plot_title, times_label, '_'+band, channel_label, method, clip_range, figsave=figsave)

    if ('all' not in channel_param) and ('per_channel' not in channel_param):
        raise ValueError(f"{channel_param} is not a valid input parameter for the VGOS skyplots. Please use 'all', or 'per_channel'")

    if obs_map: load_map(observatory_location, az_max_list, el_max_list, f'RF-disturbances for session {session}, {observatory_location}; Band {band}', f'{save_session_path(session)}/{session}_map_{times_label}_{band}', figsave=figsave)
   

def run_vgos_analysis(session, doy_beginning, end_indicator, spec, settings, add_params, method):
    # Make list of requested files for a given session, beginning & end time, which then will be loaded it´nto a list
    # Remove calibration signals when loading the dataset
    # Execute optional requests according to settings-vector (keep cal. signals, GNUplot, spectraplot, skyplot) and additional parameter-vector (GNUplot dates, skyplot clipping)

    figsave = True     # save output figures
    obs_map = True      # whether to visualize the skyplot on the map of the observatory
    files_path = get_session_path(session)

    band, freq_vector = return_frequencies_vgos(spec)   # Band & frequency parameters

    beginning = timeformat_files(doy_beginning)
    print(f'\n Fetching VGOS Data of session, Band {band}...')

   # Search for all files containing session name
    files = []
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
    if interval == []: raise ValueError('No data found for the selected date. Please input a valid timestamp for this session')
   
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
    # Remove calibration signals if settings[0] is True:
    datasets_list = []
    for i in range(len(interval)):
        f = np.loadtxt(files_path+interval[i])
        if (settings[0] is True): dataset = remove_peaks_vgos(f)
        else: dataset = f
        datasets_list.append(dataset)

    # GNU plots:
    if (settings[1] is True):
        GNU_doy = add_params[0]
        match = re.search(r'\d{4}\.\d{2}\.\d{2}\.\d{2}:\d{2}:\d{2}', GNU_doy)
        if match: 
            # Plot spectral frequencies at requested time:
            print(f' Plotting dataset at the time {GNU_doy}')
            GNU_time = timeformat_files(GNU_doy)
            gnu_index = files.index(f"{session}_ws_{GNU_time}_scansection_{spec}.spec")
            vgos_gnuplot(datasets_list[gnu_index], session, GNU_doy, band, figsave=figsave)
        elif GNU_doy == 'all': 
            # Plot all datasets
            print(f'Plotting all datasets for the selected timespan')
            for i, dataset in enumerate(datasets_list):
                GNU_doy_i = datetime.datetime.strptime(year + "-" + interval[i][10:13], "%Y-%j").strftime("%Y.%m.%d")
                GNU_doy_i += f'.{interval[i][14:16]}:{interval[i][16:18]}:00' 
                print(GNU_doy_i)
                vgos_gnuplot(dataset, session, GNU_doy_i, band, figsave=figsave)
        else:
            raise ValueError('-GNUplot argument wrong. Please use format YYYY.MM.DD.hh:mm:ss, or `all`')
        
    # Plot spectrogram, if true
    if (settings[2] is True):
        print(' Plotting spectrograms (pol H and V) of the session...')
        vgos_spectra_plot(datasets_list, session, band, freq_vector, x_axis, times_label, method, figsave=figsave)

    # Skyplot part:
    if (settings[3] is not None):
        print(f' Plotting skyplots for the session...')
        start_id = interval[0][10:18]
        azimuths, elevations = get_summary_for_session(session, start_id, 'ws', len(datasets_list))
        channel = settings[3]    # choose channel here: el.[1,16]
        clip_range = add_params[1]
        vgos_sky_plot(azimuths, elevations, datasets_list, session, band, channel, times_label, freq_vector, method, clip_range, obs_map, figsave=figsave)