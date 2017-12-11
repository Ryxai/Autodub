from pydub import AudioSegment
from librosa import load, time_to_samples
from numpy import arange


def mp3_convert(path):
    """
    Converts an input file to a wav file with the same name so that it can be
    imported by librosa, which is capable of lower level DSP operations.
    :param path: The filepath of the file to write to.
    :return: The path of the newly converted wav version of the input file
    """
    cnvt = AudioSegment.from_mp3(args.input)
    new_path = path[:-3] + "wav"
    cnvt.export(new_path, format="wav")
    return new_path


def get_audio(filepath):
    """
    Loads the audio from the given filepath via librosa and returns the
    contents and the sampling rate as a tuple.
    :param filename: A filepath from which to retrieve the contents of a wave
    file
    :return:  A tuple containing the sampling rate and the audio data as a
    numpy array in the form: data, rate
    """
    return load(filepath, sr=44100, mono=True)


def melodia_timestamps(length, sample_rate, hop_size):
    """
    Returns an array of timestamps given the length of the of array and the
    interval distance. This formulation is specific to the implementation
    of the vamp plugin toolset
    :param length: The length of the audio (number of samples)
    :param sample_rate: The sample rate for the given track (Hz)
    :param hop: The overlap distance between analysis windows
    :param intvl: The distance between each analysis window
    :return:
    """
    sample_rate = float(sample_rate)
    return 8 * (hop_size / sample_rate) + arange(length) * (
    hop_size / sample_rate)


def t_to_samples(start, stop, sample_rate):
    """
    Given a start and a stop time and the sampling rate, returns the number of
    samples that would be taken over that time interval.
    Essentially time * sample_rate
    :param start: The time the sample starts
    :param stop: The time the sample stops
    :param sample_rate: The sampling rate for the given audio
    :return: An integer that represents the number of frames given the sample
    """
    return int(time_to_samples(stop - start, sample_rate))