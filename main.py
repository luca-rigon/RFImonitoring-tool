import argparse
import time
import datetime
from datetime import timedelta
from utils import *
from load_map import load_map
from run_sx_analysis import run_sx_analysis
from run_vgos_analysis import run_vgos_analysis

if __name__ =='__main__':

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
    ap.add_argument("-d", "--doperand", required=False,
                    help="Duration of session")
    ap.add_argument("-session", "--sessoperand", required=True,
                    help="Session information")
    ap.add_argument("-type", "--typeoperand", required=True,
                    help="Antenna type: VGOS or S/X-Legacy")


    # Optional argument: Specify if the calibration signals are to be filtered out. Default=True
    ap.add_argument("-removeCal", "--caliboperand", required=False,
                    help="Visualize spectral data for one session file, for given input times")
    
    # Optional argument: Specify the method for evaluating signal outliers; can be 'max', 'mean', 'median'. Default='max'
    ap.add_argument("-method", "--methodoperand", required=False,
                    help="Analysis method")

    # Optional argument: Visualize spectral data for a single session time, or all of them. Input has to be datetime-format/'all'
    ap.add_argument("-GNUplot", "--gnuoperand", required=False,
                    help="Visualize spectral data for one session file, for given input times")
    
    # Optional argument: Visualize spectrogram of session. Only valid/true for input = 1 
    ap.add_argument("-spectrogram", "--spectroperand", required=False,
                    help="activate spectrogram-plots")
    
    # Optional argument: Create skyplot for session. Only valid/true for input = 1-16 or 'all', requesting the corresponding channel
    ap.add_argument("-skyplot", "--skyoperand", required=False,
                    help="activate skyplot")

    # Optional argument: Clip skyplot data. Only valid/true for input = 1, default: False
    ap.add_argument("-sky_clip", "--clipoperand", required=False,
                    help="activate skyplot")
    args = vars(ap.parse_args())
    

    session = args['sessoperand']
    doy_beginning = args['foperand']
    antenna_type = args['typeoperand']    # vgos or sx

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

    if antenna_type == 'vgos':
        print(f'\nAnalysis for session {session} (VGOS), start time: {doy_beginning}')

        bands = ['0', '1', '2', '3']    # A, B, C, D
        for i in range(len(bands)):
            run_vgos_analysis(session, doy_beginning, end_indicator, bands[i], params, GNU_doy, method)

    elif antenna_type == 'sx':
        print(f'\nAnalysis for session {session} (S/X-Legacy), start time: {doy_beginning}')
        
        run_sx_analysis(session, doy_beginning, end_indicator, params, GNU_doy, method)

    else:
        raise ValueError(f"{antenna_type} is not a valid antenna parameter for this analysis. Please use 'vgos', or 'sx' ")

    end_time = time.time()
    print(f'\nAnalysis Done! Total duration: {round(end_time - start_time, 3)} s \n')