import vamp
from numpy import std, average
from functools import reduce

def analyze_wav(wav, sample_rate):
    """
    Retrieves melody frequency data given an initial audio wav sample and
    frequency data. Uses the melodia vamp plugin to achieve this.
    More:
    http://www.justinsalamon.com/news/melody-extraction-in-python-with-melodia

    Note: Melodia works best with background noise and will extrapolate melody
    from background noise if applied to a song with lots of silence or acapella.

    :param wav: The wav data to be analyzed
    :param sample_rate: The sampling rate that the data was taken at (according
    to brief looks at melodia documentaion setting it to 44100 Hz is best
    practice).
    :return: An array of frequency data sampled at the standard melodia sample
    rate. This data represents the what the melodia plugin believes the
    frequency of the melody is given a polyphonic environment.

    """
    #Get melody data
    data = vamp.collect(wav, sample_rate, "mtg-melodia:melodia")
    hop, freq_data = data["vector"]
    #Add timestamps
    return freq_data

def note_threshold(sample, curr_note, threshold, weight):
    """
    Determines the bounds of a note given an array of already existing samples
    and a new sample that is used to determine whether the sample should
    be added to the current note or not. Uses the equation:
    \prod_i^n \frac{1}{w^i} * (x_i - \bar{x}/ \sigma} where \sigma is the
    standard deviation for the given note and \bar{x} is the average

    :param sample: The current sample (should be drawn directly from the wav
    analysis
    :param curr_note: An array of samples that resemble a note
    :param threshold: A chosen threshold (bounded between 0 and 1)
    :param weight: A chosen weight bounded between (0 and 1), increasingly
    far away samples should have decreasing weights
    :return: A boolean whether or not the current sample should be included
    in the current note or based on this primitive analysis belongs
    to a new note
    """
    #The threshold and weight should be determined through trial and error
    #any other method will result in kludging even worse than this monstrosity
    #is already.
    if threshold > 1 or threshold < 0:
        raise RuntimeError("Threshold weight must be > 0 and < 1, it" +
                              "currently is set to " + str(threshold))
    if weight > 0.5 or weight < 0:
        raise RuntimeError("The weight must be > 0 and < 1, it  currently" +
                           "is set to" + str(weight))
    if (len(curr_note) < 3):
        return True
    std_dev = std(curr_note)
    avg = average(curr_note)
    weighted_product = reduce(lambda x,y: x *
                                   pow(1/float(weight),y[0]) *
                                          ((y[1][0] - avg)/std_dev),
                              enumerate(curr_note),0)
    return threshold < weighted_product/float(len(curr_note))

def notate(freq_data, threshold_func, sample_rate):
    """
    Separates the frequency samples into separate notes based on the the
    passed in threshold function.

    :param wav: The analyzed frequency data, should be an array/list
    :param threshold_func: A boolean function that returns whether the
    given sample should be considered in the current note or note
    :param sample_rate: The rate at which the audio was sampled when imported
    :return: An list of note data
    """
    notes = []
    active_note = []
    for item in freq_data:
        if active_note == []:
            active_note.append(item)
        elif threshold_func(item[0], active_note):
            active_note.append(item)
        else:
            notes.append({
                "frequency": reduce(lambda acc, x: acc + x[0], active_note, 0) /
                             float(len(active_note)),
                "start": active_note[0][1],
                "end": active_note[-1][1],
                "sample rate": sample_rate
            })
            active_note = []
            active_note.append(item)
    return notes


def get_nearest_frequency_match(samples, target_fq):
    """
    Given a list of sample metadata and a target frequency, will take a
    comparison function and will sort the list of samples by preference on which
    sample best represents the target frequency.
    :param samples: A list of sample metadata to sort
    :param target_fq: The target frequency to match
    :return: The metadata tag that best represent the target frequency
    """
    eval_func = lambda x: abs(x["frequency"] - target_fq)
    return sorted(samples, key=eval_func)[0]
