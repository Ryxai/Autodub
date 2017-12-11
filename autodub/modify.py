from librosa import time_to_samples, stft, istft, phase_vocoder, resample
from scipy import hanning
from numpy import zeros
from functools import reduce

def get_wav_slices_from_metadata(wav, metadata):
    """
    Given a metadata tag about the audio, retrieves the given audio slice
    represented by the metadata tag and inserts it into the metadata under the
    key 'audio'. This data is wav data and is stored as a numpy array. This
    function is designed to work on lists of metadata tags.
    :param wav: The wav file to retrieve audio slices from
    :param metadata: A list of metadata dictionaries containing note data, that
    is data relating the notes of the audio including (most importantly
    start/stop times).
    :return: A list of updated data including the metadata as well as the sliced
    audio
    """
    out = []
    for datum in metadata:
        start = time_to_samples(datum["start"])[0]
        end = time_to_samples(datum["end"])[0]
        out.append({
            "frequency": datum["frequency"],
            "start": datum["start"],
            "end": datum["end"],
            "audio": wav[start:end]
        })
    return out

def freq_shift(wav, fq, tg_fq, sample_rate):
   """
   Shifts the given audio from its current frequency the target frequency
   :param wav: The audio to shift
   :param fq: The frequency of the current audio
   :param tg_fq: The target frequency to shift to (in Hz)
   :param sample_rate: The sampling rate used for the audio when imported
   by librosa
   :return: A shifted frequency audio sample to the given target frequency
   """
   D = stft(wav)
   shift_factor = 1 + (fq - tg_fq)/fq
   D_shifted = phase_vocoder(D, shift_factor)
   x_shifted = istft(D_shifted)
   return resample(x_shifted, sample_rate, int(sample_rate/shift_factor))

def time_shift(wav, tg_lngth):
    """
    Shifts the audio length while not affecting pitch by using a phase
    vocoder
    :param wav: The audio to shift
    :param tg_lngth: The target length of the audio to be shifted, measured
    in terms of number of samples
    :return: A timeshifted audio file that is the length of tg_length
    """
    D = stft(wav)
    shift_factor = tg_lngth/len(wav)
    D_shifted = phase_vocoder(D, shift_factor)
    return istft(D_shifted)

def concentate_samples(wav_arrs):
    """
    Concentates a list of numpy arrays into a extended array
    :param wav_arrs: A list of short wav arrays (numpy)
    :return: A concentated wav array
    """
    return reduce(lambda acc, x: acc.append(x), wav_arrs)

def concentate_samples_with_windowing(wav_arrs, hop_size):
    """
    Concentates a number of samples with a wav array, blending their ends
    together slightly.

    :param wav_arrs: The array of samples to be concentated
    :param hop_size: The amount of blend between the samples
    :return: A concentated wav array, numpy array
    """
    length = reduce(lambda acc, x: acc + len(x), wav_arrs)
    lengthened_samples = [time_shift(arr, len(arr) + 2 * hop_size)
                                                        for arr in wav_arrs]
    out_wav = zeros(length)
    curr_length = 0
    for sample in enumerate(lengthened_samples) and curr_length <= length:
        sample_length = len(sample[1]) - 2 * hop_size
        window = hanning(len(sample[1]))
        windowed_sample = window * sample[1]
        if sample[0]:
            out_wav[0: sample_length] += \
                    windowed_sample[0 + hop_size, sample_length + hop_size]
        else:
            out_wav[curr_length: curr_length + sample_length] += \
                windowed_sample[0 + hop_size, sample_length + hop_size]
    return out_wav

def generate_silence(length, sample_rate):
    """
    Generates a silent segment given a length of time and sample_rate
    :param length: Length of the silence to generate in seconds
    :param sample_rate: The sample rate for the output track
    :return: An numpy array of zeros that is the length of silence in samples
    to be used.
    """
    return zeros(time_to_samples(length, sample_rate))

