import os
import datetime


def get_session_path(session):
    # Update here for the correct session path where the path is different.
    path = f'Datasets/{session}/'
    return path


def save_session_path(session):
    # Create new folder within a Figures folder with the name of the requested session

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
    date_formatted = str(tt.tm_yday)+'-'+date[11:13]+date[14:16]
    return date_formatted


def return_band(spec):
    # Return frequency channels & their names for the chosen spectral band
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


def remove_peaks(sessionfile):
    # Remove calibration signals found at following given indices
    # Input - sessionfile: One single dataset-file from the session, converted into an array

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


def get_summary_for_session(session, start_id, length_of_list): 
    # Extract relevant session information (timestamp, azimuth and elevation values) from the corresponding summary file, provided in the 'session_name_sum.txt' - format
    # Inputs: - session: active session_name
    #         - start_id: timestamp-id corresponding to the chosen start-time of the analysis 
    #         - length_of_list: number of registered points to be considered, according to session duration

    with open(f'{get_session_path(session)}/{session}ws_sum.txt', 'r') as file:
        lines = file.readlines()
        timestamps = []
        azimuth_values = []
        elevation_values = []
        
        for line in lines[14:]:     # Start line iteration onl after header
            values = line.split()
            
            if len(values) >= 7:
                # Check if the line has enough columns
                if values[3] != 'Az' and values[4] != 'El':
                    # Extract azimuth and elevation values
                    time = values[0]
                    azimuth = float(values[3])
                    elevation = float(values[4])
                    timestamps.append(time)
                    azimuth_values.append(azimuth)
                    elevation_values.append(elevation)

    # Extract the time range of interest:
    start_index = timestamps.index(start_id)
    end_index = start_index + length_of_list
    return azimuth_values[start_index:end_index], elevation_values[start_index:end_index]