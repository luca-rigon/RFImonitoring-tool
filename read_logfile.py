import re

def return_frequencies_sx(path_to_file):

    print(' Individuated channels: Starting Frequency, Bandwidth:')
    freq_vector = []
    band_flags = []
    not_found = ['None']
    value = 0.0 
    value_prev = '' 

    with open(path_to_file, 'r') as file:
        lines = file.readlines()

    for i in range(1,15):
        channel = str(i)
        if len(channel) == 1: channel = '0' + channel
        value_str = ''
        rf_type = ''
        bandwidth = ''
        get_frequency = False

        # Define a regular expression pattern to match the line containing 'bbc0i' & the frequency shift value
        pattern = r'\d{4}\.\d{3}\.\d{2}:\d{2}:\d{2}\.\d{2}&dbbc\w+/bbc' + channel + r'=(.*),(\w+),(\d+\.\d+)'
        pattern_ = r'\d{4}\.\d{3}\.\d{2}:\d{2}:\d{2}\.\d{2}&ifdsx/lo=lo' + rf_type +',(\d+\.\d+),.*'

        # Search for the 'bbc0i' line and extract the information
        for line in lines:
            match = re.search(pattern, line)
            if match:
                # Extract Freq. values
                value_str = match.group(1)
                rf_type = match.group(2)
                bandwidth = match.group(3)
                get_frequency = True

            elif get_frequency:
                # Shift frequencies
                pattern_ = r'\d{4}\.\d{3}\.\d{2}:\d{2}:\d{2}\.\d{2}&ifdsx/lo=lo' + rf_type +',(\d+\.\d+),.*'
                match_ = re.search(pattern_, line)
                if match_:
                    shift_freq = match_.group(1)
                    if i == 1 or i == 8: value = float(shift_freq) - float(value_str)     # at band 1 and 8: 8080 - frequenz
                    else: value = float(shift_freq) + float(value_str)
                    value_str = str(value)
                    
                    print(f"  'BBC{channel}': {value} MHz, {float(bandwidth)} MHz")
                    break  

        if i == 1:
            freq_vector.append(('', value_str))
        else:
            freq_vector.append((value_prev, value_str))
            if i == 14: 
                if get_frequency: freq_vector.append((str(value + float(bandwidth)), ''))
                else:   freq_vector.append(('', ''))

        if get_frequency: 
            value_prev = str(value + float(bandwidth))
            band_flags.append(True)
        else:
            value_prev = ''
            band_flags.append(False)
            not_found.append(f'BBC{channel}')

    if len(not_found) == 1: print('  All channels assigned')
    else: print(f'  Not assigned channels: {not_found[1:]}')
    # Add information for the lower polarized bands (1 & 8) at their respective channel positions (9 & 10):
    band_flags = band_flags[:8] + [band_flags[0]] + [band_flags[7]] + band_flags[8:]
    return freq_vector, band_flags


# session = 'i24026'
# path_to_file = f'Datasets/{session}/{session}wz.log'
# freq_vector, band_flags = return_frequencies_sx(path_to_file)
# print(freq_vector, band_flags)
