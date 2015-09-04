"""
Build microwave pulse sequences for an arbitrary waveform generator in order to perform diamond NV-center experiments.

This module constructs waveforms for use with an arbitrary waveform generator (hereafter referred to as an AWG)
in performing microwave manipulation of electronic state populations in negatively-charged diamond NV centers.
It relies on AWG.py, a separate module containing functions and constants related to interfacing with a particular AWG.

In short, two channels contain sinusoidal signals in the radio-frequency (RF) range of differing phase, later used with
an I-Q mixer to modulate a microwave source into pulses. Also controls a laser, photon counter, and reference counter.

Diagram of the waveforms:
                                                     _____________________________________________________________
laser:      ________________________________________|                            prepare
                                                     ________________
counter:    ________________________________________|<-measure_time->|____________________________________________
                                                                                           ________________
reference:  ______________________________________________________________________________|                |______
                                                                                          |<---reference_offset-->|
RF pulse:   ____________/\  /\  /\  /\  __________________________________________________________________________
                          \/  \/  \/  \/
            |  pre_RF  |    RF_window   |  post_RF  |

RF_window may contain any pulse sequence defined in the global dictionary pulse_dict. These sequences are designed to
influence and investigate magnetic spin properties of the NV center, examples being Rabi, Ramsey, and Hahn-echo pulses.

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import AWG

pulse_dict = {'Rabi':   ['pulse_width'],
              'Ramsey': ['pi/2', 'tau', 'pi/2'],
              'Hahn':   ['pi/2', 'tau', 'pi', 'tau', 'pi/2']}
"""Dictionary holding structural definitions of different types of pulses.

Keys are names of pulse types, values are lists outlining the structure for the sequence implied by that pulse type.
These lists represent alternating segments of microwave pulse and microwave silence for a given sequence.
They may contain the following terms to define the length associated with each segment: pulse_width, pi/2, pi, tau

"""


def _chp_sine(points, phase=0.0):
    """Produce a 'sine' wave as [0,1,0,-1] repeating for the given number of points, and using the given phase.

    This in an internal function, and is automatically implemented if proper conditions are met on a call to sine().

    Args:
        points (int):   Number of points to include in the resulting wave
        phase (float):  Phase of the wave in degrees - any non-multiple of 90 will be rounded to the nearest

    Returns:
        An array of unsigned 16-bit ints of length points of the 'sinusoidal' wave scaled for the AWG memory

    """
    # Round up the number of cycles to use, and include an extra as well
    # Always need at least 3 points past end to ensure 0-3 phase shift possible
    cycle = np.tile([0, 1, 0, -1], np.ceil((points / 4) + 1))

    # Convert phase to number of points to shift indices
    # 1 yields cos, 2 yields -sin, 3 yields -cos
    offset = int(np.round((phase % 360) / 90))
    start = offset % 4  # Constrain offset to [0,3]
    stop = start + points - 1  # First point is not included while counting
    result = cycle[start:stop + 1]  # Indexing is upper-bound exclusive

    return AWG.rescale(result)


def sine(points, sample_rate, freq, phase=0.0):
    """Calculate an array of sine values with the specified number of points, sample rate, frequency, and phase.

    If sample_rate is not a harmonic of freq, there will be aliasing in the amplitude of the resulting wave - an error
    will be raised rather than continuing.

    Args:
        points:      Number of x-values for which to calculate a result
        sample_rate: Sampling rate to use in spacing the x-values
        freq:        Frequency of the desired wave
        phase:       Phase offset of the desired wave in degrees

    Returns:
        An array of unsigned 16-bit ints of length points scaled for the AWG memory

    """
    # If sample rate is fourth harmonic of freq, can use cheaper function
    if sample_rate == 4 * freq:
        return _chp_sine(points, phase)

    # Raise an error if amplitude will vary
    elif sample_rate % freq:
        raise ValueError('Sample rate not divisible by freq: aliasing expected in amplitude of resulting waveform')

    else:
        dt = 1 / sample_rate                                # Time between points in s
        result = np.arange(0, points, dtype=np.float64)     # Array of [0, (points-1)]
        np.multiply(result, dt, out=result)     # Multiply for true time values [0, dt, 2*dt,..., (points-1)*dt]
        w = 2*np.pi * freq                      # Pre-calculate omega
        phase = np.deg2rad(phase)               # Convert phase to radians

        np.sin((w * result - phase), out=result)  # Calculate sine for each time value

        return AWG.rescale(result)


def cosine(points, sample_rate, freq, phase=0.0):
    return sine(points, sample_rate, freq, phase + 90)


def pulse(wfm_len, start, stop):
    """Create a waveform of the given length consisting of all zeros except ones from start to stop.

    Args:
        wfm_len (int):  The number of points in the waveform to return
        start (int):    The index at which to start the pulse - first point in the pulse
        stop (int):     The index of the point immediately after the pulse ends

    Returns:
        An array of unsigned 16-bit integers full of zeros except ones from wfm[start] to wfm[stop - 1]

    """
    wfm = np.zeros(wfm_len, dtype=np.uint16)
    wfm[start:stop] = 1

    return wfm


def get_timing(pulse_type, dt, prepare, pre_RF, post_RF, pi_2, pulse_width, tau):
    """Convert given waveform and pulse timing values into numbers of waveform points for each section.

    Args:
        pulse_type:     A pulse type defined in pulse_dict
        dt:             Time between waveform points in ns
        prepare:        prepare in ns
        pre_RF:         pre_RF in ns
        post_RF:        post_RF in ns
        pi_2:           pi_2 in ns
        pulse_width:    pulse_width in ns
        tau:            tau in ns

    Returns:
        timing:      An array of prepare, pre_RF, RF_window, post_RF converted to integral multiples of dt
        RF_sequence: An array of the pulse sequence timing values converted to integral multiples of dt

    """
    global pulse_dict
    RF_sequence = []
    # For each entry in the sequence for the given pulse type, put the corresponding value in a list.
    for item in pulse_dict[pulse_type]:
        if item == 'pulse_width':
            RF_sequence.append(pulse_width)
        elif item == 'pi/2':
            RF_sequence.append(pi_2)
        elif item == 'tau':
            RF_sequence.append(tau)
        elif item == 'pi':
            RF_sequence.append(pi_2 * 2)
        else:
            raise ValueError('Invalid keyword found in definition of {} pulse type: {}'.format(pulse_type, item))

    # Put in an array and convert values to number of waveform points for each sequence section
    RF_sequence = np.array(RF_sequence)
    RF_sequence = np.divide(RF_sequence, dt, out=RF_sequence).round().astype(np.uint32)  # Integral multiples of dt
    RF_window = RF_sequence.sum() + 1  # Number of points in the entire RF window, inclusive of start point

    # Convert timing values into number of waveform points
    timing = np.array([prepare, pre_RF, 0, post_RF])
    np.divide(timing, dt, out=timing)  # Put timings into multiples of dt
    timing = timing.round().astype(np.uint32)  # Round to nearest integer multiples of dt
    timing[2] = RF_window  # Added later to avoid dividing by dt twice

    return timing, RF_sequence


def generate_RF(wfm_len, dt, RF_start, RF_freq, RF_sequence):
    """Build waveforms for channels 1 and 2 using the specified pulse type and parameters.

    Builds the pulse sequence part of the waveform based on the pre-computed RF_sequence array from get_timing.

    Args:
        wfm_len:        Length of the entire waveform in points
        dt:             Time between waveform points in s
        RF_start:       Waveform index of the first point in the RF sequence
        RF_freq:        Frequency of the RF sine and cosine signals to use when RF is on
        RF_sequence:    Array of the timings of the pulse sequence in numbers of waveform points, inclusive of RF_start

    Returns:
        I_wfm: Waveform for channel 1 including pulses
        Q_wfm: Waveform for channel 2 including pulses

    """
    dt /= 10**9  # Put dt in s
    RF_period = (1 / RF_freq) / dt  # Fractional multiples of dt in the RF period

    # Arrays of full waveform length
    I_wfm = np.full(wfm_len, AWG.ARB_ZERO, dtype=np.uint16)
    Q_wfm = np.full(wfm_len, AWG.ARB_ZERO, dtype=np.uint16)

    phase_start = RF_start
    start = RF_start
    stop = start
    for index, entry in enumerate(RF_sequence):
        stop += entry  # Shift stop over by the number of points in the new entry

        if (index + 1) % 2:  # For every odd entry (pulse)
            # Number of fractional wavelengths at start is correction to keep in phase with other pulses of same phase
            phase_corr = (((start - phase_start) % RF_period) / RF_period) * 360

            # Note: slicing arrays is upper-bound exclusive, need stop + 1 to function properly
            I_wfm[start:stop] = sine(entry, AWG.samp_clk_freq, RF_freq, phase_corr)
            Q_wfm[start:stop] = cosine(entry, AWG.samp_clk_freq, RF_freq, phase_corr)
        else:  # For every even entry (pause)
            pass

        start += entry  # Move start over by the number of points in the old entry

    return I_wfm, Q_wfm


def interpret_pulse(pulse_type, *params):
    """Interpret given pulse type and parameters into a full waveform representation.

    Args:
        pulse_type: Key in pulse_dict for the type of pulse to construct using the given parameters.
        *params: Parameter list containing the following:
            params[0] = prepare             Amount of time the laser is preparing the zero state
            params[1] = pre_RF              Delay after the laser turns off but before the MW pulses start
            params[2] = post_RF             Delay after the MW pulses end, but before the laser turns on
            params[3] = measure_time        Length of time to open the counter gate for
            params[4] = measure_offset      Offset time of the counter and RF with respect to the laser turning on
            params[5] = reference_offset    Offset time of the reference pulse with respect to the laser turning off
            params[6] = RF_freq             Frequency in MHz of the sine and cosine waves in the RF pulses
            params[7] = pi_2                Pi/2 pulse time for Ramsey, Hahn, etc.
            params[8] = pulse_width         Pulse width to use for Rabi
            params[9] = tau                 Delay time between pulses in Ramsey, Hahn, etc.

            Pulse parameters have been placed at the end of the list to facilitate future expansion if necessary.

    Returns:
        A list containing the following, used to send to the AWG, or graph the waveform:
            encoded_channel_1:  Waveform for Channel 1, ready to upload to the AWG
            encoded_channel_2:  Waveform for Channel 2, ready to upload to the AWG
            I_wfm:              The RF pulse used for Channel 1
            Q_wfm:              The RF pulse used for Channel 2
            laser_pulse:        The laser pulse used
            measure_pulse:      The measure pulse used
            period:             The period of the resulting waveform, used to count for a certain number of periods to
                                avoid drift in counts due to fewer/more laser pulses counted in a certain length of time

    """
    dt = (10 ** 9) / AWG.samp_clk_freq  # Conversion from clock frequency to time between waveform points in ns

    # Initialize all variables (which are given in ns) including any corrections
    prepare = params[0]
    pre_RF = params[1]
    post_RF = params[2]
    measure_time = params[3] / dt       # Fractional multiples of dt to measure for
    measure_offset = params[4] / dt     # Fractional multiples of dt (and direction) to offset the measure pulse
    reference_offset = params[5] / dt   # Fractional multiples of dt to offset the reference pulse before laser-off
    # Pulse parameters
    RF_freq = params[6] * 10**6  # Accept frequency in MHz and convert to Hz
    pi_2 = params[7]
    pulse_width = params[8]
    tau = params[9]

    # Calculate waveform length
    timing, RF_sequence = get_timing(pulse_type, dt, prepare, pre_RF, post_RF, pi_2, pulse_width, tau)
    prepare, pre_RF, RF_window, post_RF = timing  # Multiples of dt, except RF_window which is number of points
    total_points = timing.sum()  # Total number of points in the adjusted waveform

    # Min length is 384; exceptions are handled in AWG.py, but for convenience here, anything less is set to 384
    if total_points < 384:
        wfm_len = 384

    # Length must be a multiple of 32
    elif total_points % 32:  # If not a multiple, (32 - total % 32) is number of points to next multiple
        wfm_len = total_points + (32 - (total_points % 32))  # Set to next multiple of 32
        prepare += (wfm_len - total_points)  # Put the difference into prepare, since it is least timing-sensitive
        """Note: This would be implied anyway, since prepare simply goes until the end of the waveform, but is
        included here to more explicitly demonstrate what is changing.
        """
    else:
        wfm_len = total_points  # If already a multiple of 32, do not change

    # Laser pulse
    laser_start = pre_RF + RF_window - 1 + post_RF # Fractions of dt, position along waveform
    laser_stop = wfm_len  # Stops at end of wfm
    laser_start, laser_stop = AWG.correct_marker_edges(wfm_len, laser_start, laser_stop)  # Correct positions
    laser_pulse = pulse(wfm_len, laser_start, laser_stop)  # Create laser pulse

    # Measure counter gate
    measure_start = laser_start + measure_offset  # Fractional position of measure start including offset
    measure_stop = measure_start + measure_time  # Fractional position of measure stop based on width
    measure_start, measure_stop = AWG.correct_marker_edges(wfm_len, measure_start, measure_stop)  # Correct positions
    measure_pulse = pulse(wfm_len, measure_start, measure_stop)  # Create measurement pulse

    # Reference counter gate
    if reference_offset < 0:  # If negative reference offset given, implement - otherwise, set to zeros
        reference_start = laser_stop + reference_offset  # Fractional position; reference_offset expected to be negative
        reference_stop = reference_start + measure_time  # Fractional position
        reference_start, reference_stop = AWG.correct_marker_edges(wfm_len, reference_start, reference_stop)  # Correct
        reference_pulse = pulse(wfm_len, reference_start, reference_stop)  # Create reference pulse
    else:
        reference_pulse = np.zeros(wfm_len, dtype=np.uint16)

    # Create RF pulse, including corrections to laser and measure placement
    RF_start = int(np.round(pre_RF + measure_offset))
    I_wfm, Q_wfm = generate_RF(wfm_len, dt, RF_start, RF_freq, RF_sequence)

    # Waveforms for unused markers will be zeros
    zero = np.zeros(wfm_len, dtype=np.uint16)

    # At this point the desired waveforms are ready to be encoded with their markers
    encoded_channel_1 = AWG.encode(I_wfm, laser_pulse, measure_pulse)
    encoded_channel_2 = AWG.encode(Q_wfm, zero, reference_pulse)

    # Return the encoded waveforms for the AWG, as well as the finished parts to graph them
    return [encoded_channel_1, encoded_channel_2, I_wfm, Q_wfm, laser_pulse, measure_pulse, wfm_len * dt * 10**-9, np.round(measure_offset)]


def counter_offset_test(*params):
    dt = (10 ** 9) / AWG.samp_clk_freq  # Conversion from clock frequency to time between waveform points in ns

    # In multiples of dt
    laser_time = int(np.round(params[0] / dt))  # Round to nearest multiple of dt
    measure_time = params[3] / dt
    measure_offset = params[4] / dt

    total_points = 6 * laser_time + 1  # include first point

    # Waveform length must be a multiple of 32
    if total_points % 32:  # If not a multiple, (32 - total % 32) is number of points to next multiple
        wfm_len = total_points + (32 - (total_points % 32))  # Set to next multiple of 32
        # Note: this increase will simply be stuck on the end after the third laser pulse
    else:
        wfm_len = total_points  # If already a multiple of 32, do not change

    laser = np.zeros(wfm_len, dtype=np.uint16)
    measure = np.zeros(wfm_len, dtype=np.uint16)
    zero = np.zeros(wfm_len, dtype=np.uint16)

    # +1's are included since accessing arrays is upper-bound exclusive
    laser[:laser_time + 1] = 1
    laser[(2 * laser_time):(3 * laser_time) + 1] = 1
    laser[(4 * laser_time):(5 * laser_time) + 1] = 1

    measure_start, measure_stop = AWG.correct_marker_edges(wfm_len, measure_offset, measure_offset + measure_time)
    measure[measure_start:measure_stop + 1] = 1

    encoded_channel_1 = AWG.encode(zero, laser, measure)
    encoded_channel_2 = AWG.encode(zero, zero, zero)

    # Return the encoded channels to send to the AWG, as well as the calculated waveform parts for graphing purposes
    return [encoded_channel_1, encoded_channel_2, laser, measure, wfm_len * dt * 10**-9 - 1]


def continuous_wave(*params):
    """Generate a continuous sine (ch 1) and cosine (ch 2) signal - used during laser focus adjustments.

    Args:
        RF_freq: Frequency of the desired RF signal in MHz

    Returns:
        A list of

    """
    RF_freq = params[6] * 10**6  # Accept frequency in MHz and convert to Hz
    wfm_len = 32000  # An arbitrary multiple of 32 - made large to lessen stitching if edges are uneven

    # Calculate sine and cosine waves
    I_wfm = sine(wfm_len, AWG.samp_clk_freq, RF_freq)
    Q_wfm = cosine(wfm_len, AWG.samp_clk_freq, RF_freq)

    # Dummy waveforms for unused markers
    zero = np.zeros(wfm_len)
    one = np.ones(wfm_len)

    # Encode into final waveform
    encoded_channel_1 = AWG.encode(I_wfm, one, one)
    encoded_channel_2 = AWG.encode(Q_wfm, zero, zero)

    return [encoded_channel_1, encoded_channel_2, I_wfm, Q_wfm, one, zero]


def plot_wfm(pulse_type, include_corrections=False, *params):
    """Display a graph of the pulse resulting from the given parameters and instrument adjustments in AWG.py."""

    if pulse_type == 'Continuous':
        results = continuous_wave(*params)
        measure_offset = 0
    else:
        results = interpret_pulse(pulse_type, *params)
        measure_offset = results[7]

    # Rescale waveforms
    I_wfm = AWG.rescale(results[2])
    Q_wfm = AWG.rescale(results[3])
    # Can rescale waveform back, or graph what was scaled for AWG instead

    # If flag is set, apply instrument corrections
    if include_corrections:
        I_wfm = I_wfm * AWG.CH_1_AMP + AWG.CH_1_DC_OFFSET
        Q_wfm = Q_wfm * AWG.CH_2_AMP + AWG.CH_2_DC_OFFSET

    # Show two waveform periods on the graph
    I_wfm = np.concatenate((I_wfm, I_wfm))
    Q_wfm = np.concatenate((Q_wfm, Q_wfm))
    sent_laser_pulse = np.concatenate((results[4], results[4]))
    measure_pulse = np.concatenate((results[5], results[5]))

    # Add measure offset (AOM delay) to the laser pulse when graphed
    resulting_laser_pulse = np.insert(sent_laser_pulse, 0, np.zeros(measure_offset, dtype=np.uint16))
    resulting_laser_pulse = resulting_laser_pulse[:sent_laser_pulse.size]  # Shorten to same length to fit on graph

    dt = (10 ** 9) / AWG.samp_clk_freq  # Time between waveform points in ns
    t = np.arange(0, I_wfm.size, step=1, dtype=np.float64)  # upper-bound exclusive, yields correct number of points
    t = np.multiply(t, dt)  # Put time domain in ns

    # Plot with two subplots, one the RF waveforms, one the laser and marker pulses
    plt.subplot(211)
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.plot(t, I_wfm, 'r-')
    plt.plot(t, Q_wfm, 'b-')
    plt.legend(['Channel 1', 'Channel 2'], loc='lower right', fontsize='small')
    plt.axis([0, t[-1], -2, 2])

    plt.subplot(212)
    plt.xlabel('Time (ns)')
    plt.ylabel('Voltage (V)')
    plt.plot(t, measure_pulse, 'b-')
    plt.plot(t, resulting_laser_pulse, 'g-')
    plt.plot(t, sent_laser_pulse, 'g--')
    plt.legend(['Measure Pulse', 'Resulting Laser Pulse', 'Sent Laser Pulse'], loc='lower right', fontsize='small')
    plt.axis([0, t[-1], -0.05, 1.05])

    plt.legend()
    plt.show()

    #return period


def parse_commands():
    """Parse the command line arguments and carry out the given instructions."""

    # Specify the argument framework
    parser = argparse.ArgumentParser(description='Create and upload microwave pulse sequences to the AWG.')
    parser.add_argument('-initialize', '-init', action='store_true',
                        help='reset and initialize the settings of the AWG for programming in arbitrary mode')
    parser.add_argument('-mode', help='specify the run mode of the program',
                        choices=['test', 'Continuous'] + list(pulse_dict.keys()))  # Include all keys from pulse_dict
    parser.add_argument('-params', type=float, nargs='+', help='gathers parameters for the chosen pulse type')
    parser.add_argument('-plot', '-graph', action='store_true', help='display a graph of the resulting waveform')

    # Parse the given arguments using the argument framework above
    commands = parser.parse_args()

    # Sort through which action to take:
    if commands.initialize:
        AWG.initialize()
        # exit here?

    if commands.mode:  # If mode is specified
        if commands.mode == 'test':
            # Run test code here
            pass

        elif commands.mode == 'Continuous':
            if commands.params:
                if commands.graph:
                    plot_wfm(commands.mode, True, *commands.params)

                else:  # Upload continuous wave
                    result = continuous_wave(*commands.params)
                    ch1 = result[0]
                    ch2 = result[1]
                    period = result[6]
                    AWG.refresh_waveforms(ch1, ch2)
                    print(period)  # Return period of the pulse to LabView to measure for a certain number of pulses

            else:
                raise ValueError('Parameters not specified')

        elif commands.params:  # If mode is a pulse type and parameters are given:
            if commands.graph:
                plot_wfm(commands.mode, True, *commands.params)

            else:  # Upload pulse waveform
                result = interpret_pulse(commands.mode, *commands.params)
                ch1 = result[0]
                ch2 = result[1]
                period = result[6]
                AWG.refresh_waveforms(ch1, ch2)
                print(period)  # Return period of the pulse to LabView to measure for a certain number of pulses
    else:
        raise ValueError('Mode not specified')


def main():
    parse_commands()
    return 0

# Only run if this script is invoked directly, so it may be imported without any issues
if __name__ == '__main__':
    status = main()
    argparse.sys.exit(status)
