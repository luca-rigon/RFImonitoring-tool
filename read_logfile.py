import re

def return_frequencies_sx(path_to_file):

    print(' Individuated channels: Starting Frequency, Bandwidth:')
    freq_vector = []
    band_flags = []
    not_found = ['None']
    value = 0.0 
    value_prev = '' 
    bits_str = ''

    with open(path_to_file, 'r') as file:
        lines = file.readlines()

    date_pattern = r'\d{4}\.\d{3}\.\d{2}:\d{2}:\d{2}\.\d{2}'

    ## Bedingung form=geo --> es werden X1l & X8l genützt; wenn nicht geo --> Bänder 15 u. 16
    ## Diese Annahme muss geprüft werden
    form_pattern = date_pattern + r'&setupsx/form=(\w+)'
    is_geo = True
    for line in lines:
        match_form = re.search(form_pattern, line)
        if match_form:
            if match_form.group(1) != 'geo': is_geo = False
            break

    # Find bit mask within log file - indicates BBC-allocations
    setbits_pattern = date_pattern + r'&setupsx/mk5c_mode=vdif,([0-9a-fA-F]x[0-9a-fA-F]+),,(\d+\.\d+)'
    for line in lines:
        match_bits = re.search(setbits_pattern, line)
        if match_bits:
            bits_str = match_bits.group(1)

            for i in range(len(bits_str)):
                bit = bits_str[-(i+1)]
                if bit == '0':
                    band_flags += [0, 0]
                elif bit == 'x':
                    if is_geo==False and bits_str[0]!='0':  # Remove X1l and X8l - slots, add BBC 15 & 16 in the end
                        band_flags =  band_flags[:8] + band_flags[10:] + [2, 2]
                    break
                else:
                    band_flags += [1, 1]
            break

    band_shift = 0

    for i in range(1,17):
        channel = str(i)
        if len(channel)==1: channel = '0' + channel
        value_str = ''
        rf_type = ''
        bandwidth = ''
        bbc_is_set = False      # Sanity check
        
        # Search for the 'bbc0i' line and extract the information
        bbc_pattern = date_pattern + r'&dbbc\w+/bbc' + channel + r'=(.*),(\w+),(\d+\.\d+)'
        for line in lines:
            match = re.search(bbc_pattern, line)
            if match:
                value_str = match.group(1)
                rf_type = match.group(2)
                bandwidth = match.group(3)
                bbc_is_set = True

            elif bbc_is_set:
                # Define a regular expression pattern to match the line containing & the frequency shift value
                freq_pattern = date_pattern + r'&ifdsx/lo=lo' + rf_type + r',(\d+\.\d+),.*'
                freq_match = re.search(freq_pattern, line)
                if freq_match:
                    shift_freq = freq_match.group(1)
                    if i == 1 or i == 8: value = float(shift_freq) - float(value_str)     # at band 1 and 8: 8080 - frequenz
                    else: value = float(shift_freq) + float(value_str)
                    value_str = str(value)
                    
                    print(f"  'BBC{channel}': {value} MHz, {float(bandwidth)} MHz")
                    break  
        
        if i == 1:
            freq_vector.append(('', value_str))
        else:
            freq_vector.append((value_prev, value_str))

        if is_geo == True and i == 9: band_shift += 2

        if bbc_is_set: 
            value_prev = str(value + float(bandwidth))
            if i<15 and band_flags[i+band_shift-1] != 1: band_flags[i+band_shift-1] = 1
        else:
            value_prev = ''
            if i<15 and band_flags[i+band_shift-1] != 0: band_flags[i+band_shift-1] = 0
            not_found.append(f'BBC{channel}')
        
    freq_vector.append((value_prev, ''))

    if len(not_found) == 1: print('  All channels assigned')
    else: print(f'  Not assigned channels: {not_found[1:]}')

    # Remove empty allocations:
    freq_vector_out = [('', '')]
    k = 0
    j = 0
    while k < len(freq_vector):
        if freq_vector[k][1] != '': 
            freqs = list(freq_vector_out[j])
            freqs[1] = freq_vector[k][1]
            freq_vector_out[j] = tuple(freqs)
            freq_vector_out.append((freq_vector[k+1][0], ''))
            j += 1
        k += 1
    
    return freq_vector_out, band_flags


# session = 'q24027'
# path_to_file = f'Datasets/{session}wz.log'
# freq_vector, band_flags = return_frequencies_sx(path_to_file)
# print(freq_vector, band_flags)
