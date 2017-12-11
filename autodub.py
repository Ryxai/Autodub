import argparse, re
from autodub import *
from librosa import output
from functools import reduce, partial

if __name__=="__main__":
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("input", type=str, help="path to an input file")
        parser.add_argument("output", nargs="?", type=str, help="output path")
        parser.add_argument("--dub", "-d", type=str, help="path(s) to file(s)",
                            nargs="*")
        parser.add_argument("--weight", "-w", type=float, default=0.5,
                            help="A weight used to help determine the length " +
                                 "of note")
        parser.add_argument("--threshold", "-t", type=float, default=0.4,
                            help="A thresholdto determine the length of a note")
        parser.add_argument("--random", "-r", action="store_true",
                            help="choose samples by random rather than by" +
                                 " the closest frequency match")
        parser.add_argument("--blend", "-b", type=float, default=0,
                            help="amount to blend the the dub track with the " +
                                 "original audio")
        return parser.parse_args()
    args = parse_args()

    #Handle mp3 input files
    mp3_matcher = re.compile(".*\.mp3")
    if mp3_matcher.match(args.input):
        utility.mp3_convert(args.input)

    threshold_func = lambda samp, note: note_threshold(samp, note,
                                                    args.threshold, args.weight)

    melodia_sample_rate = 44100
    melodia_hop_size = 128
    #Get wav data
    melody_wav, melody_rate = get_audio(args.input)
    dubs_wavs = [get_audio(x) for x in args.dub]
    dubs_len = len(dubs_wavs)
    #Analyze melody
    melody_freqs = analyze_wav(melody_wav, melody_rate)
    dub_freqs = [analyze_wav(x[0], x[1]) for x in dubs_wavs]
    #Get timestamps
    melody_stamps = melodia_timestamps(len(melody_freqs), melodia_sample_rate,
                                                               melodia_hop_size)
    dub_stamps = [melodia_timestamps(len(dub), melodia_sample_rate,
                                     melodia_hop_size) for dub in dub_freqs]
    #Folding melody and timestamp data together
    mel_demarcated_fqs = zip(melody_freqs, melody_stamps)
    dubs_demarcated_fqs = [zip(dub_freqs[i],dub_stamps[i]) for i
                                                            in range(dubs_len)]
    #Extract note data
    melody_notes = notate(mel_demarcated_fqs, threshold_func, melody_rate)
    dub_notes = [notate(dubs_demarcated_fqs[i],threshold_func, dubs_wavs[i][1])
                            for i in range(dubs_len)]


    #Get wav slices from audio slices from metadata for the dub tracks
    dub_notes = [get_wav_slices_from_metadata(dubs_wavs[i], dub_notes[i])
                                                       for i in range(dubs_len)]

    #Handle dub prioritization here in future iteration
    dubs = reduce(lambda acc,x: acc + x, dub_notes)
    #Adding a silence node for negative frequencies

    #Shift frequency and clip to proper length
    final_dub_samples = []
    for note in melody_notes:
        tg_fq = note["frequency"]
        tgt_samples = t_to_samples(note["start"],
                                               note["end"], note["sample rate"])
        if tg_fq < 0:
            shifted_dub = generate_silence(note["end"] - note["start"],
                                                            note["sample rate"])
        else:
            dub = get_nearest_frequency_match(dubs, tg_fq)
            shifted_dub = freq_shift(dub["audio"],dub["frequency"],
                                                    tg_fq, note["sample rate"])
        final_dub_samples.append(time_shift(shifted_dub, tgt_samples))

    # Put together final wav array
    final_wav = concentate_samples_with_windowing(final_dub_samples, 12)

    #Blend if blending option is active
    if args.blend != 0:
        if args.blend > 1 or args.blend < 0:
            print("Cannot blend with a factor exceeding range, ignoring")
        else:
            final_wav *= args.blend
            melody_wav += melody_wav * (1 - args.blend)
    #Output
    if args.output:
        output.write_wav(args.output, final_wav, melody_rate)
    else:
        for x in final_wav:
            print(x)
