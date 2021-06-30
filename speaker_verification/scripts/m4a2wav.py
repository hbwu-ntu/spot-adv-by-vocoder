#!/usr/bin/env python
# encoding: utf-8

import os
import torch
import argparse
import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool

def findAllSeqs(dirName,
                extension='m4a',
                load_data_list=False,
                speaker_level=1):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers
        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index
        outSpeakers
        The speaker labels (in order)
    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension
    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.
    """

    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    print("finding {}, Waiting...".format(extension))
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root, filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]

    for key, index in speakersTarget.items():
        outSpeakers[index] = key

    print("find {} speakers".format(len(outSpeakers)))
    print("find {} utterance".format(len(outSequences)))

    return outSequences, outSpeakers


def m4a2wav_main(m4a_path):
    wav_path = m4a_path.strip(".m4a") + ".wav"
    if os.path.exists(wav_path):
        os.remove(wav_path)
    cmd = "ffmpeg -v 8 -i {} -f wav -acodec pcm_s16le {}".format(m4a_path, wav_path)
    os.system(cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', help='dataset dir', type=str, default="data")
    args = parser.parse_args()

    outSequences, outSpeakers = findAllSeqs(args.dataset_dir,
                load_data_list=False,
                speaker_level=1)

    outSequences = np.array(outSequences, dtype=str)
    utt_paths = outSequences.T[1].tolist()

    with Pool(16) as p:
        p.map(m4a2wav_main, utt_paths)


