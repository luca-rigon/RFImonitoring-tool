import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import contextily as cx

def get_session_path(session):
    # Update here for the correct session path where the path is different.
    path = f'Datasets/{session}/'
    return path

def save_session_path(session):
    # Create new folder within a Figures folder with the name of the requested session - All generated figures are saved here
    results_folder = os.path.join(".", "Figures")
    os.makedirs(results_folder, exist_ok=True)
    path = os.path.join(results_folder, session)
    os.makedirs(path, exist_ok=True)
    return path

def timeformat_files(date):
    # Start and End format which is read from the user as an input
    # Convert given datetime-string (YYYY.MM.DD:hh:mm:ss) into the DOY format so that it can be read as a filename

    format = "%Y.%m.%d.%H:%M:%S"    
    dt = datetime.datetime.strptime(date, format)
    tt = dt.timetuple()

    # Control the given day & hour format to prevent type errors
    day = str(tt.tm_yday)
    hours = str(tt.tm_hour)    
    mins = str(tt.tm_min)
    if tt.tm_yday<10:
        day = '00'+day
    elif tt.tm_yday<100:
        day = '0'+day
    if tt.tm_hour<10:
        hours='0'+hours
    if tt.tm_min<10:
        mins='0'+mins

    date_formatted = day+'-'+hours+mins
    return date_formatted

def sxL_IndexToChannel(i):
    # Defined only for i==8 -> CH1 or i==9 -> CH8
    # For correct S/X-Plot labeling
    if i == 8: return '1'
    else: return '8'


def return_frequencies_vgos(spec):
    # Return frequency channels & their names for the chosen spectral band (VGOS)
    # Fixed frequencies & bandwidths (32 MHz)
    # Input - spec: value {0,1,2,3} which corresponds to the active spectral band

    band = ''
    freq_vector = []
    if spec == '0': 
        band = 'A'      #           ch 1                   ch 2                ch 3                   ch 4                  ch 5                  ch 6                  ch 7                  ch 8
        freq_vector = [('', '3032.4'), ('3064.4', '3064.4'), ('3096.4', '3096.4'), ('3128.4', '3224.4'), ('3256.4', '3320.4'), ('3352.4', '3384.4'), ('3416.4', '3448.4'), ('3480.4', '3480.4'), ('3512.4', '')]
    elif spec == '1': 
        band = 'B'
        freq_vector = [('', '5272.4'), ('5304.4', '5304.4'), ('5336.4', '5336.4'), ('5368.4', '5464.4'), ('5496.4', '5560.4'), ('5592.4', '5624.4'), ('5656.4', '5688.4'), ('5720.4', '5720.4'), ('5752.4', '')]
    elif spec == '2': 
        band = 'C'
        freq_vector = [('', '6392.4'), ('6424.4', '6424.4'), ('6456.4', '6456.4'), ('6488.4', '6584.4'), ('6616.4', '6680.4'), ('6712.4', '6744.4'), ('6776.4', '6808.4'), ('6840.4', '6840.4'), ('6872.4', '')]
    elif spec == '3': 
        band = 'D'
        freq_vector = [('', '10232.4'), ('10264.4', '10264.4'), ('10296.4', '10296.4'), ('10328.4', '10424.4'), ('10456.4', '10520.4'), ('10552.4', '10584.4'), ('10616.4', '10648.4'), ('10680.4', '10680.4'), ('10712.4', '')]
    return band, freq_vector


def remove_peaks_vgos(sessionfile):
    # Remove calibration signals found at following given indices (VGOS-files)
    # Usually fixed, check regularly for the correct locations
    # Input - sessionfile: One single dataset-file from the session, converted into a npy-array

    shape = sessionfile.shape   # (6400, 17)
    sessionfile_filtered = sessionfile
    # Indices for calibration bands to eliminate:
    calibration_indices = (
                        [80, 1080, 2080, 3080, 4080, 5080, 6080],       # ch1+9
                        [680, 1680, 2680, 3680, 4680, 5680],            # ch2+10
                        [880, 1880, 2880, 3880, 4880, 5880],            # ch3+11
                        [80, 1080, 2080, 3080, 4080, 5080, 6080],       # ch4+12
                        [880, 1880, 2880, 3880, 4880, 5880],            # ch5+13
                        [280, 1280, 2280, 3280, 4280, 5280, 6280],      # ch6+14
                        [880, 1880, 2880, 3880, 4880, 5880],            # ch7+15
                        [480, 1480, 2480, 3480, 4480, 5480]             # ch8+16
                        )
    
    # Remove the peaks for the given indices:
    for k in range(1,shape[1]):
        calibration_peaks = calibration_indices[(k-1)%8]
        for p in calibration_peaks:
            sessionfile_filtered[p,k] = (sessionfile[p-1,k] + sessionfile[p+1,k]) / 2 # Replace peak with values close to it
    return sessionfile_filtered


def remove_peaks_sx(sessionfile, lower=True):
    # Remove calibration signals found at following given indices (S/X-files)
    # Usually fixed, check regularly for the correct locations; different calibration when lower channels are not allocated
    # Input - sessionfile: One single dataset-file from the session, converted into a npy-array

    shape = sessionfile.shape   # (6400, 17)
    sessionfile_filtered = sessionfile
    # Indices for calibration bands to eliminate:
    calibration_indices = (
                        [505, 1005, 1505, 2005, 2505, 3005, 3505],         # upper channels
                        [495, 995, 1495, 1995, 2495, 2995, 3495],          # lower channels
                        [125, 625, 1125, 1625, 2125, 2625, 3125, 3625]     # all channels, if lower=False
                        )
    
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


def get_summary_for_session(session, start_id, suffix, length_of_list): 
    # Extract relevant session information (timestamp, azimuth and elevation values) from the corresponding summary file, provided in the 'session_name_sum.txt' - format
    # Inputs: - session: active session_name
    #         - start_id: timestamp-id corresponding to the chosen start-time of the analysis 
    #         - length_of_list: number of registered points to be considered, according to session duration

    with open(f'{get_session_path(session)}{session}{suffix}_sum.txt', 'r') as file:
        lines = file.readlines()
        timestamps = []
        azimuth_values = []
        elevation_values = []
        
        for line in lines[16:]:     # Start line iteration onl after header
            values = line.split()
            if len(values) >= 9:
                # Check if the line has enough columns
                if values[3] != 'Az' and values[4] != 'El':
                    # Extract azimuth and elevation values
                    time = values[0]
                    azimuth = float(values[3])
                    elevation = float(values[4])
                    timestamps.append(time[:8])
                    azimuth_values.append(azimuth)
                    elevation_values.append(elevation)

    # Extract the time range of interest:
    start_index = timestamps.index(start_id)
    end_index = start_index + length_of_list
    return azimuth_values[start_index:end_index], elevation_values[start_index:end_index]


def return_polar_values(azimuths, elevations, abins, rbins, channel_values):
    # Returns the RFI-values in the horizontal coordinate system, binned onto the unit circle, as well as the azimuth & elevation values of the highest disturbance
    # Inputs: - azimuth, elevation & the corresponding channel values (1D arrays) in the same indexed order
    #         - binning of a unit circle (abins, rbins) of a certain resolution (60x30)

    values = np.zeros((len(abins), len(rbins)))
    n_samples = len(channel_values)

    for i in range(n_samples):
        az_index = int(round(azimuths[i]/6)) 
        el_index = int(round(elevations[i]/3)) 
        if az_index == 60: az_index = 0      # close circle
        # Only consider the highest value at each position:
        if channel_values[i] > values[az_index, el_index]:
            values[az_index, el_index] = channel_values[i]

    # Get the directions towards the highest value within chosen channel:
    max_index = np.unravel_index(np.argmax(values, axis=None), values.shape)
    az_max = np.radians(max_index[0]*6 + 3)
    el_max = max_index[1]*3 #+ 2
        
    return values, az_max, el_max


def polar_plot(session, theta, R, values, az_max_ch, el_max_ch, plot_title, times_label, band, channel_label, method, clip_range, figsave=True):
    # (Polar) Plot of the previously found values in the horizontal coordinate system
    # Plot arrow pointing towards highest disturbance (at az_max_ch, el_max_ch)
    
    clip_label = ''
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

    # Clipping settings - normalize the colors onto the specified range; easier comparison
    if (clip_range is not None):
        clip_min, clip_max = clip_range  
        cax.set_clim(vmin=clip_min, vmax=clip_max)
        if values.max()==clip_min or values.min()==clip_max: plot_title += f'\nAll values outside the clipping-range [{clip_min},{clip_max}]'
        clip_label = f'_clip_{clip_min}-{clip_max}'

    fig.colorbar(cax)
    fig.suptitle(plot_title)
    if figsave: 
        plt.savefig(f'{save_session_path(session)}/{session}_skyplot_{times_label}{band}{channel_label}_{method}{clip_label}.png', dpi=300)
        plt.close()
    else:
        plt.show()


def load_map(session, location, azimuth, elevation, savepath, figsave=True):
    # Visualize map (taken from OSM) of the observatory location with drawn directions towards the main disturbances
    # Inputs: - location name: default='Wettzell, Bad Kotzting'
    #         - azimuth (radians), elevation (degrees): lists

    place = cx.Place(location, source=cx.providers.Thunderforest.Neighbourhood(apikey='29bd4f3cbd794bdfb03426605c9d98fd'), zoom=13)
    map_image = place.im
    (width, height, _) = map_image.shape
    scale = np.min([width,height])/2
    delta = scale/3

    # Convert azimuth, elevation to cartesian coordinates (row,column):
    r = elevation*scale/90          # scale radius: ->>> consider zoom-factor
    theta = azimuth - np.pi/2
    row = r*np.cos(theta)
    col = r*np.sin(theta)

    # image center:
    row_0 = (height - 1) / 2
    col_0 = (width - 1) / 2

    # Plot map, arrows & labels:
    plt.figure(figsize=(8, 8))
    plt.imshow(map_image) #[127:384,:]
    plt.arrow(row_0-delta, col_0, 2*delta, 0, width=0.5, head_width=0.5)
    plt.arrow(row_0, col_0-delta, 0, 2*delta, width=0.5, head_width=0.5)
    plt.annotate('N', (row_0-4.5, col_0-delta-4))
    plt.annotate('S', (row_0-4.5, col_0+delta+15))
    plt.annotate('W', (row_0-delta-15, col_0+4.5))
    plt.annotate('E', (row_0+delta+4, col_0+4.5))
    plt.arrow(row_0, col_0, row, col, width=2.5, head_width=8, color='red')
    plt.annotate(str(round(azimuth*180/np.pi)) + '°', (row_0+row+8, col_0+col+8), bbox=dict(fc='white', edgecolor='white', alpha=0.5), color='red')
    plt.axis('off')
    plt.title(f'RF-disturbances for session {session}, {location}')
    plt.text(0.05, 0.95, '© Thunderforest, © OpenStreetMap contributors', fontsize=8, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
    if figsave: 
            plt.savefig(f'{save_session_path(session)}/{savepath}.png', dpi=300)
            plt.close()
    else:
        plt.show()