"""
Helper library to upload custom arbitrary waveforms to a Tabor Electronics arbitrary waveform generator using PyVISA.

'Arbitrary waveform generator' is hereafter shortened to AWG.

"""

import numpy as np
import visa

# Useful debugging info can be printed to the console with the following PyVISA command
# visa.log_to_screen()

# Instrument constants
INSTR_ID = 'USB0::0x168C::0x218A::0000214340::INSTR'
MAX_ARB_MEMORY = 16000000  # Maximum number of points in the memory for each channel (32M with opt. 1)
MAX_SEGMENTS_PER_CHANNEL = 16000  # Maximum number of segments per channel
MIN_CLK_FREQ = 10e6
MAX_CLK_FREQ = 2.3e9
ARB_MAX = 16383  # Highest arbitrary point value, yields full scale amplitude (whatever voltage that is set to)
ARB_ZERO = 8192  # Arbitrary point value corresponding to 0 V output

# Correction values determined by oscilloscope inspection
# Amplitudes and offsets are in volts
CH_SKEW = -120  # Channel skew of channel 2 with respect to channel 1 in picoseconds
CH_1_AMP = 0.950
CH_1_DC_OFFSET = 0.005
CH_2_AMP = 1
CH_2_DC_OFFSET = 0.003

# Global variables
AWG_error_status = ''  # Global container for AWG error status codes
samp_clk_freq = 2.0E9  # Sample clock frequency to set


def initialize():
    """Reset the AWG and prepare it for programming in arbitrary mode using predetermined best settings."""

    # Create an instance of the VISA resource manager and open AWG as a resource.
    rm = visa.ResourceManager()
    instr = rm.open_resource(INSTR_ID, send_end=True, write_termination='\n', read_termination='\n', query_delay=2.0)  # instr refers to the AWG throughout

    # Reset and wait until operation completion
    instr.write('*RST')
    instr.write('*CLS')
    instr.query('*OPC?')

    # Set general parameters (those which apply to both channels)
    instr.write(':inst:sel 1')  # Set active channel to 1
    instr.write(':freq:rast {}'.format(samp_clk_freq))  # Set arbitrary sample clock frequency
    instr.write(':inst:coup:state on;:inst:coup:skew {}E-12'.format(CH_SKEW))  # Couple and set Ch 2 skew to best value

    # Set channel-specific parameters:
    active_channel = 1
    while active_channel <= 2:  # Loop over both channels
        instr.write(':inst:sel {}'.format(active_channel))  # Set active channel
        instr.write(':outp off;:marker1:state off;:marker2:state off')  # Turn all outputs off

        # Set to arbitrary mode, clear all sequences, and delete segment table
        instr.write(':func:mode user;:seq:del:all;:trac:del:all')
        instr.write(':init:cont on')  # Run in continuous mode

        # Set default marker widths to 0; they will be programmed along with the arbitrary waveform
        # Set marker high and low levels
        instr.write(':marker1:width 0;:marker1:volt:high 1;:marker1:volt:low 0')
        instr.write(':marker2:width 0;:marker2:volt:high 1;:marker2:volt:low 0')

        # Set Ch 1 and 2 amplitude and DC offset to corrective values
        if active_channel == 1:
            instr.write(':volt {};:volt:offs {}'.format(CH_1_AMP, CH_1_DC_OFFSET))
        elif active_channel == 2:
            instr.write(':volt {};:volt:offs {}'.format(CH_2_AMP, CH_2_DC_OFFSET))

        active_channel += 1

    # If there was an error, raise an exception detailing what it was.
    err = instr.query('syst:err?')
    if err[0] != '0':
        raise RuntimeError('AWG reports {}'.format(err))


def rescale(wfm):
    """Scale waveforms back and forth between representation in the range [-1,1] and [0,ARB_MAX].

    If it is not already unsigned 16-bit ints, the wave will be converted to such and scaled for the AWG.
    If already unsigned 16-bit ints, the wave will be converted to 64-bit floats and scaled to [-1,1].
    """

    # If not uint16, assume it is in [-1,1] and convert and scale to AWG range
    if wfm.dtype != 'uint16':
        scaled_wfm = np.multiply(wfm, ARB_ZERO)  # Scale to range (if out is wfm, original data is changed)
        np.round(scaled_wfm, out=scaled_wfm)  # Round values to nearest integers
        scaled_wfm = scaled_wfm.astype(np.uint16) + ARB_ZERO  # Convert type and offset to ARB_ZERO
        np.clip(scaled_wfm, 0, ARB_MAX, out=scaled_wfm)  # Clip values to proper range

    # If already uint16, assume it is in arbitrary representation and convert and scale to [-1,1]
    # Note: Accuracy will be limited by integer bit depth
    else:
        scaled_wfm = wfm.astype(np.float64) - ARB_ZERO  # Convert to floats and remove zero offset
        np.divide(scaled_wfm, ARB_ZERO, out=scaled_wfm)  # Divide by full-scale amplitude to scale to [-1,1]

    return scaled_wfm


def correct_marker_edges(wfm_len, start, stop=None):
    """Correct marker edge placement to multiples of 4, while retaining accuracy to the greatest extent possible.

    Resolution of marker outputs is limited to four waveform points; thus, marker points may only be placed at
    waveform points which are divisible by 4.
    Start and stop refer to the position along the waveform in fractions of dt that the marker edges are supposed to be.
    Start is the first point in the pulse, stop is the first point after the pulse
    Values of 0 or wfm_len for start or stop respectively will handle edge cases.

    Last marker point will always be 3 indices earlier than wfm end (wfm_len - 4)
    If stop is near but not equal to wfm_len, better to put early than lose edge entirely if rounded up to wfm_len
    If stop is exactly wfm_len, this will not apply and the marker will loop over to the next period
    Takes fractional or integer values of start, stop

    """
    if wfm_len <= 0 or wfm_len % 32:
        raise ValueError('Invalid wfm_len')
    elif start < 0 or start >= wfm_len:
        raise ValueError('Marker start index out of bounds')

    if stop is None:  # If stop not specified, only correct start
        if wfm_len - start < 4:  # If near the end of the wfm
            start = wfm_len - 4  # Set to final marker point in wfm

        start_mod_4 = start % 4

        if start_mod_4 < 2:  # If closer to the lesser multiple of 4
            start -= start_mod_4

        elif start_mod_4 == 2:  # If directly between multiples of 4
            start -= start_mod_4  # Could also be +=, introduces some systematic error

        elif start_mod_4 > 2:  # If closer to the greater multiple of 4
            start += 4 - start_mod_4

        return int(start)

    else:  # if stop specified
        if stop <= 0 or stop > wfm_len:
            raise ValueError('Marker stop index out of bounds')
        elif stop <= start:
            raise ValueError('Marker stop index less than or equal to marker start index')

        # If near the end of the wfm, set to final marker point in wfm
        if wfm_len - start < 4:
            start = wfm_len - 4
        if 0 < (wfm_len - stop) < 4:
            # If stop is equal to wfm_len, this will not apply and the marker will continue for the full wfm length
            stop = wfm_len - 4

        start_mod_4 = start % 4
        stop_mod_4 = stop % 4

        if start_mod_4 < 2:
            start -= start_mod_4
            if stop_mod_4 < 2:
                stop -= stop_mod_4
            elif stop_mod_4 == 2:
                stop -= stop_mod_4
            elif stop_mod_4 > 2:
                stop += 4 - stop_mod_4

        elif start_mod_4 == 2:
            if stop_mod_4 < 2:
                start -= start_mod_4
                stop -= stop_mod_4
            elif stop_mod_4 == 2:
                start -= start_mod_4  # Could also be += for both, introduces some systematic error
                stop -= stop_mod_4
            elif stop_mod_4 > 2:
                start += 4 - start_mod_4
                stop += 4 - stop_mod_4

        elif start_mod_4 > 2:
            start += 4 - start_mod_4
            if stop_mod_4 < 2:
                stop -= stop_mod_4
            elif stop_mod_4 == 2:
                stop += 4 - stop_mod_4
            elif stop_mod_4 > 2:
                stop += 4 - stop_mod_4

        if stop == start:  # If a very short pulse results in stop = start, set width to at least 4 points
            stop += 4

        return int(start), int(stop)


def marker_encode(marker_shape, marker_num):
    """Encode the data for a single marker into the proper bits for representation in the final waveform.

    Marker data is stored in the 15th and 16th bits in the last 8 16-bit words in each 32-word group of data.
    They are sampled every 4 clock cycles, and offset along the regular waveform data by 24 points due to hardware.

    Args:
        marker_shape (ndarray): The desired shape of the marker output, stored in an array as 0's or 1's
        marker_num (int):       Which marker to encode this shape as, marker 1 or marker 2

    Returns:
        An array of 16-bit unsigned ints with the given marker encoded inside
        In decimal representation, marker 1 should be either 0 or 16,384 (2^14); marker 2 should be 0 or 32,768 (2^15)
        This dtype allows combination with waveform data by addition.

    """

    marker_shape = np.clip(marker_shape, 0, 1)  # Clip array to be in [0,1]

    if not issubclass(marker_shape.dtype.type, np.integer):  # If array is not of integer type
        np.ceil(marker_shape)  # Round any decimals to 1

    marker_num += 13  # Starting at 1, bits will need to be shifted either 14 or 15 places depending on which marker
    encoded_marker = np.zeros(marker_shape.size, dtype=np.uint16)

    for i in np.arange(0, marker_shape.size, 32):  # Every 32 points along marker_shape,
        for j in np.arange(0, 8):  # for 8 consecutive points,
            encoded_marker[i + 24 + j] = marker_shape[i + 4 * j]  # Set encoded_marker to every fourth marker_shape
            encoded_marker[i + 24 + j] <<= marker_num  # Shift data to the correct bit in these waveform points

    return encoded_marker


def encode(wfm, marker_1, marker_2):
    """Combine waveform and marker data into a final waveform ready to upload to the AWG.

    Markers are stored in bits 15 and 16 (marker 1 and 2) in the last 8 16-bit words in each 32-word group of data.
    They are sampled every 4 clock cycles, and offset along the regular waveform data by 24 points due to hardware.
    All inputs are arrays of full waveform length.

    Args:
        wfm (ndarray):      Waveform data in range [0,16383]
        marker_1 (ndarray): The desired shape of marker 1 output, stored as either 0's or 1's in an array
        marker_2 (ndarray): The desired shape of marker 2 output, stored as either 0's or 1's in an array

    Returns:
        An array of full waveform length with both markers encoded into the proper bits along the waveform

    """

    if wfm.dtype != 'uint16':
        raise Exception('Waveform data-type is not np.uint16')

    # Add the parts together to create the final waveform
    markers = marker_encode(marker_1, 1) + marker_encode(marker_2, 2)
    np.add(wfm, markers, out=markers)

    # The result is stored in markers because the dtype is known to be correct - wfm dtype may not be
    return markers


def upload_wfm(wfm, segment_ID=1):
    """Upload one encoded waveform to the AWG in the specified segment number in the current channel.

    Args:
        wfm: encoded waveform data in a numpy array of unsigned 16-bit ints, 384 <= length < MAX_MEMORY_PER_CHANNEL
        segment_ID: arbitrary memory segment number to define and upload the waveform to; defaults to 1
        # Number of bytes the waveform data takes up (2 bytes per 14-bit waveform point)
    """

    # Create an instance of the VISA resource manager and open Tabor AWG on USB as a resource.
    rm = visa.ResourceManager()
    instr = rm.open_resource(INSTR_ID, send_end=True, write_termination='\n', read_termination='\n', query_delay=0.5)

    # Calculate header info
    num_of_digits = int(np.floor(np.log10(wfm.nbytes)) + 1)  # Number of base-10 digits needed to express num_of_bytes

    # Define and select a new memory segment for the waveform
    instr.write(':trac:def {},{};:trac:sel {}'.format(segment_ID, wfm.size, segment_ID))

    # If the array is big-endian (>), swap to little-endian (<)
    if wfm.dtype.byteorder == '>u2':
        wfm.byteswap(inplace=True).newbyteorder(new_order='<')

    # Send the waveform to the AWG in two parts: first the command string, then the binary data
    # If a corrupt command is sent, the AWG will lock up and power will need to be cycled to clear it
    instr.write(':trac#{}{}'.format(num_of_digits, wfm.nbytes))
    instr.write_raw(wfm.tobytes() + '\n'.encode('utf8'))  # Terminate binary transfer with newline
    instr.query('*OPC?')  # Check operation complete bit after upload


def refresh_waveforms(ch1, ch2):
    """ Clear the waveforms currently uploaded to each channel, then upload new given waveforms in their place.

    Args:
        ch1: Waveform to upload to channel 1
        ch2: Waveform to upload to channel 2

    """
    # Create an instance of the VISA resource manager and open Tabor AWG on USB as a resource.
    rm = visa.ResourceManager()
    instr = rm.open_resource(INSTR_ID, send_end=True, write_termination='\n', read_termination='\n', query_delay=0.5)

    # Upload channel 1
    instr.write(':inst:sel 1')
    instr.write(':outp off;:marker1:state off;:marker2:state off')
    upload_wfm(ch1)
    instr.write(':outp on;:marker1:state on;:marker2:state on')

    # Upload channel 2
    instr.write(':inst:sel 2')
    instr.write(':outp off;:marker1:state off;:marker2:state off')
    upload_wfm(ch2)
    instr.write(':outp on;:marker1:state on;:marker2:state on')
