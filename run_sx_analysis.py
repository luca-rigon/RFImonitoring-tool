import glob
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from utils import *
from read_logfile import return_frequencies_sx

# Constants:
observatory_location = 'Wettzell, Bad Kotzting'
# Here, change the split number if needed(e.g too much files exist for processing)
split_num=160

# Plots:
def sx_gnuplot(dataset, session, time, band_flags, figsave=True):
    # Generate GNU-plot of one dataset, for a given session (Legacy S/X)
    # shape: variable, based on nr. of channel allocations (4000 x (1+xxx))
    # When all channels are used:
    # channel 0: frequencies (x-axis)
    # channels 1-8: X bands, upper
    # channels 9-10 X bands, lower (1&8)
    # channels 11-16: S bands, upper (channel name shifted down by 2: 9-14)
    # If no lower bands allocated: S bands from 9-16

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
                else:     band = 'S'
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
    if figsave:
        plt.savefig(f'{save_session_path(session)}/{session}_gnuplot_cal_{timeformat_files(time)}.png')
        plt.close()
    else:
        plt.show()

def sx_spectra_plot(datasets_list, session, band_flags, freq_vector, x_axis, times_label, method, figsave=True):
    # Plot spectrograms for a given session over all selected datasets, listed in datasets_list, for all found frequencies
    # Inputs: - band_flags: list of flags representing channel allocations
    #         - freq_vector: list of frequency ranges for allocated channels, in the required format for plotting
    #         - x_axis: x-axis label, in the required format for plotting
    #         - times_label: session duration, as label for title
    #         - method: statistical method to apply on data. Default: np.max()

    titles = ['X-band', 'S-band']
    all_channels_listed = []
    values_list = [] 
    number_channels = len([x for _, x in enumerate(band_flags) if x!=0])

    clip_min = 0
    clip_max = 3  

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

    norm = colors.Normalize(vmin=clip_min, vmax=clip_max) # Normalize the data for colorbar

    freq_vector = freq_vector[:x_count] + [(freq_vector[x_count][0], '')] + [('', freq_vector[x_count][1])] + freq_vector[x_count+1:] # Format for plotting

    # generate 2 Figures: X & S
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

        axs[(band_counts[bnd]-1)//2].set_ylabel('MHz')  # Put ylabel at ab. the middle of the axis
        plt.xlabel('Timestamps')
        cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])    # Create a shared colorbar      
        plt.colorbar(ci, cax=cax)
        fig.suptitle(f'Session {session}, {titles[bnd]}, upper polarization')
        if figsave:
            plt.savefig(f'{save_session_path(session)}/{session}_spectraplot_{times_label}_{titles[bnd]}_{method}.png', bbox_inches='tight', dpi=300)
            plt.close()
        else:
            plt.show()


def sx_sky_plot(session, azimuths, elevations, datasets_list, channel_param, times_label, band_flags, freq_vector, method, clip_range, obs_map=False, figsave=True):
    # Visualize skyplot of the datasets_list - files for given channels and corresponding lists of azimuth and elevation data
    # Inputs: - channel_param: skyplot type chosen for visualization. Can be either 'per_channel', 'per_band' to split them into bands (Xu,Su,Xl) or 'all' for all taken together
    #         - times_label: session duration, as label for title
    #         - band_flags: list of flags representing channel allocations
    #         - freq_vector: list of frequency ranges within the band, in the required format for plotting
    #         - method: statistical method to apply on data. Default: np.max()
    #         - clip_skyplot: wheter to clip the measurements onto a specific range or not
    #         - obs_map: additional parameter, if the main disturbances are to be shown on the map for a given location (Set parameter on top)

    number_channels = len([x for _, x in enumerate(band_flags) if x!=0])
    n_samples = len(datasets_list)
    method_func = getattr(np, method)       # Define function np.max(), np.mean(), np.median()
    
    # Clipping settings:
    clipping = False
    clip_min, clip_max = [0,1000]
    if (clip_range is not None):
        clip_min, clip_max = clip_range    
        clipping = True

    # Compute max/mean/median for all frequencies:
    values_channels_all = np.zeros((number_channels,n_samples))
    for ch in range(number_channels):
        for i in range(n_samples):
            values_channels_all[ch,i] = method_func(datasets_list[i][:,ch+1])

    lower_count = 0         
    if band_flags[-1] != 2 and band_flags[8] == 1: lower_count = 2      # lower band allocations
    x_count = 0
    s_count = 0

    for i in range(16):
            if band_flags[i]:
                if i < 8: x_count += 1
                else: 
                    if lower_count == 2 and i==14: break
                    s_count += 1

    # Get the direction of the highest disturbances (i.e. get indices for highest values):
    disturbance_index = np.unravel_index(np.argmax(values_channels_all, axis=None), (number_channels,n_samples))
    az_max = np.radians(azimuths[disturbance_index[1]])
    el_max = elevations[disturbance_index[1]]
    print(f' Highest disturbance {values_channels_all.max()} at ({azimuths[disturbance_index[1]]}, {elevations[disturbance_index[1]]}) - frequency channel {disturbance_index[0]+1}: {freq_vector[disturbance_index[0]-lower_count][1]} - {freq_vector[disturbance_index[0]-lower_count+1][0]}')

    # define binning
    abins = np.linspace(0,2*np.pi, 60)     # azimuth - angle; 1/6
    rbins = np.linspace(0,90, 31)          # elevation - radius; 1/3
    theta, R = np.meshgrid(abins, rbins)

    if 'all' in channel_param:
        # Run analysis over all channels
        channel_values = np.zeros(n_samples)
        for i in range(n_samples):
            channel_values[i] = method_func(values_channels_all[:,i])       # Get max/mean/median over all channels for each sample
        values, az_max_ch, el_max_ch = return_polar_values(azimuths, elevations, abins, rbins, channel_values)
        if clipping: values = np.clip(values, a_min=clip_min, a_max=clip_max)

        plot_title = f'Skyplot for {session}, over all frequencies (S and X bands): \n {freq_vector[x_count][1]} - {freq_vector[x_count][0]} MHz'
        polar_plot(session, theta, R, values, az_max_ch, el_max_ch, plot_title, times_label, '', '_all', method, clip_range, figsave=figsave)

    
    if 'per_band' in channel_param:
        # Run analysis over whole X/S band, lower/upper
        X_values = np.zeros(n_samples)
        S_values = np.zeros(n_samples)
        lower_values = np.zeros(n_samples)

        for i in range(n_samples):
            X_values[i] = method_func(values_channels_all[:x_count,i])               # Get max/mean/median over all channels for each sample
            S_values[i] = method_func(values_channels_all[x_count+lower_count:,i])       
            if lower_count == 2: lower_values[i] = method_func(values_channels_all[x_count:x_count+2,i])

        vals_per_band = [X_values, S_values]
        title_labels = ['X-bands, upper polarization', 'S-bands, upper polarization', 'X-bands, lower polarization']
        title_freqs = [f'{freq_vector[0][1]} - {freq_vector[x_count][0]}', f'{freq_vector[x_count][1]} - {freq_vector[x_count+s_count][0]}', f'{freq_vector[x_count-1][1]} - {freq_vector[1][0]}']  # NOT correct; remove completely?
        band_labels = ['Xu', 'Su', 'Xl']
        if lower_count == 2: vals_per_band.append(lower_values)      

        for bnd, band_values in enumerate(vals_per_band):
            values, az_max_bnd, el_max_bnd = return_polar_values(azimuths, elevations, abins, rbins, band_values)
            if clipping: values = np.clip(values, a_min=clip_min, a_max=clip_max)
            
            plot_title = f'Skyplot for {session}, {title_labels[bnd]}: \n {title_freqs[bnd]} MHz'
            polar_plot(session, theta, R, values, az_max_bnd, el_max_bnd, plot_title, times_label, '_bands_'+band_labels[bnd], '', method, clip_range, figsave=figsave)
    

    if 'per_channel' in channel_param:
        # Run analysis for every single channel separately
        channel_list = np.arange(1, number_channels+1)
        channel_state = 1
            
        for channel in channel_list:
            plot_title = ''
            # Adjust band labels:
            k = channel_state-1
            while band_flags[k] == 0: 
                channel_state += 1
                k += 1
                
            if channel_state < 9: 
                ch_label = f'X-{channel_state}u'
                freq_interval = f'{freq_vector[channel-1][1]} - {freq_vector[channel][0]}'
            elif lower_count == 2 and channel_state < 11: 
                ch_label = f'X-{sxL_IndexToChannel(channel_state-1)}l'
                if channel_state == 9: freq_interval = f'{freq_vector[0][1]} - {freq_vector[1][0]}'
                if channel_state == 10: freq_interval = f'{freq_vector[x_count-1][1]} - {freq_vector[x_count][0]}'
            else: 
                ch_label = f'S-{channel_state-2}u'
                freq_interval = f'{freq_vector[channel-lower_count-1][1]} - {freq_vector[channel-lower_count][0]}'

            channel_state += 1
            channel_values = values_channels_all[channel-1, :]
            values, az_max_ch, el_max_ch = return_polar_values(azimuths, elevations, abins, rbins, channel_values)
            if clipping: values = np.clip(values, a_min=clip_min, a_max=clip_max)
            
            plot_title = f'Skyplot for {session}, channel {ch_label}: \n {freq_interval} MHz'
            polar_plot(session, theta, R, values, az_max_ch, el_max_ch, plot_title, times_label, '', '_'+ch_label, method, clip_range, figsave=figsave)

    if ('all' not in channel_param) and ('per_band' not in channel_param) and ('per_channel' not in channel_param):
        raise ValueError(f"{channel_param} is not a valid input parameter for the S/X skyplots. Please use 'all', 'per_band' or 'per_channel'")

    if obs_map: load_map(session, observatory_location, az_max, el_max, f'{session}_map_{times_label}', figsave=figsave)   


def run_sx_analysis(session, doy_beginning, end_indicator, settings, add_params, method):
    # Make list of requested files for a given session, beginning & end time, which then will be loaded it´nto a list
    # Remove calibration signals when loading the dataset
    # Execute optional requests according to settings-vector (keep cal. signals, GNUplot, spectraplot, skyplot) and additional parameter-vector (GNUplot dates, skyplot clipping)

    figsave = True    # save output figures
    obs_map = True    # whether to visualize the skyplot on the map of the observatory
    files_path = get_session_path(session)
    
    freq_vector, band_flags = return_frequencies_sx(files_path+f'{session}wz.log')      # Allocation and frequencies parameter

    beginning = timeformat_files(doy_beginning)
    print(f'\n Fetching Legacy S/X Data of session ...')

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
    if interval == []: raise ValueError('No data found for the selected date. Please input a valid timestamp for this session')
   
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
    # Remove calibration signals if settings[0] is True:
    datasets_list = []
    for i in range(len(interval)):
        f = np.loadtxt(files_path+interval[i])
        if (settings[0] is True): dataset = remove_peaks_sx(f, lower=band_flags[8]) 
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
            gnu_index = files.index(f"{session}_wz_{GNU_time}_spec.out")
            sx_gnuplot(datasets_list[gnu_index], session, GNU_doy, band_flags, figsave=figsave)
        elif GNU_doy == 'all': 
            # Plot all datasets
            print(f'Plotting all datasets for the selected timespan: {year}DOY{interval[0][10:13]}, {interval[0][14:16]}:{interval[0][16:18]}:00 - {interval[-1][14:16]}:{interval[-1][16:18]}:00')
            for i, dataset in enumerate(datasets_list):
                GNU_doy_i = datetime.datetime.strptime(year + "-" + interval[i][10:13], "%Y-%j").strftime("%Y.%m.%d")
                GNU_doy_i += f'.{interval[i][14:16]}:{interval[i][16:18]}:00' 
                sx_gnuplot(dataset, session, GNU_doy_i, band_flags, figsave=figsave)
        else:
            raise ValueError('-GNUplot argument wrong. Please use format YYYY.MM.DD.hh:mm:ss, or `all`')
        
    # Plot spectrogram, if true
    if (settings[2] is True):
        print(' Plotting spectrograms (X-band, S-band) of the session...')
        sx_spectra_plot(datasets_list, session, band_flags, freq_vector, x_axis, times_label, method, figsave=figsave)

    # Skyplot part:
    if (settings[3] is not None):
        print(f' Plotting skyplots for the session...')
        start_id = interval[0][10:18]
        azimuths, elevations = get_summary_for_session(session, start_id, 'wz', len(datasets_list))
        channel = settings[3]    # choose channel here: el.[1,16]
        clip_range = add_params[1]
        sx_sky_plot(session, azimuths, elevations, datasets_list, channel, times_label, band_flags, freq_vector, method, clip_range, obs_map, figsave=figsave)