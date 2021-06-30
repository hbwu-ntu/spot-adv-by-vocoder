#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
import random
import os
import threading
from scipy import signal
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader

def round_down(num, divisor):
    return num - (num%divisor)

def loadWAV(filename, max_frames, evalmode=False, num_eval=10):
    '''
    Remark! we will set max_frames=0 for evaluation.
    If max_frames=0, then the returned feat is a whole utterance.
    '''
    # Maximum audio length
    max_audio = max_frames * 160 + 240

    # Read wav file and convert to torch tensor
    sample_rate, audio  = wavfile.read(filename)

    audiosize = audio.shape[0]

    # padding
    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        startframe = np.linspace(0, audiosize-max_audio, num=num_eval)
    else:
        startframe = np.array([np.int64(random.random()*(audiosize-max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf)+max_audio])
    feat = np.stack(feats, axis=0).astype(float)
    return feat


class AugmentWAV(object):
    def __init__(self, musan_data_list_path, rirs_data_list_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_audio = max_frames * 160 + 240
        self.noisetypes = ['noise', 'speech', 'music']
        self.noisesnr = {'noise':[0,15], 'speech':[13,20], 'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,7], 'music':[1,1]}
        self.noiselist = {}

        df = pd.read_csv(musan_data_list_path)
        augment_files = df["utt_paths"].values
        augment_types = df["speaker_name"].values
        for idx, file in enumerate(augment_files):
            if not augment_types[idx] in self.noiselist:
                self.noiselist[augment_types[idx]] = []
            self.noiselist[augment_types[idx]].append(file)
        df = pd.read_csv(rirs_data_list_path)
        self.rirs_files = df["utt_paths"].values

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2)+1e-4)
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2)+1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        audio = np.sum(np.concatenate(noises,axis=0), axis=0, keepdims=True) + audio
        return audio.astype(np.int16).astype(float)

    def reverberate(self, audio):
        rirs_file = random.choice(self.rirs_files)
        fs, rirs = wavfile.read(rirs_file)
        rirs = np.expand_dims(rirs.astype(float), 0)
        rirs = rirs / np.sqrt(np.sum(rirs**2))
        if rirs.ndim == audio.ndim:
            audio = signal.convolve(audio, rirs, mode='full')[:,:self.max_audio]
        return audio.astype(np.int16).astype(float)


class Train_Dataset(Dataset):
    def __init__(self, data_list_path, augment, musan_list_path, rirs_list_path, max_frames):

        # load data list
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        print("Train Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Train Dataset load {} utterance".format(len(self.data_list)))

        if augment:
            self.augment_wav = AugmentWAV(musan_list_path, rirs_list_path, max_frames=max_frames)

        self.max_frames = max_frames
        self.augment = augment

        self.label_dict = {}
        for idx, speaker_label in enumerate(self.data_label):
            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = [];
            self.label_dict[speaker_label].append(idx)

    def __getitem__(self, indices):
        feat = []
        for index in indices:
            audio = loadWAV(self.data_list[index], self.max_frames)
            if self.augment:
                augtype = random.randint(0,4)
                if augtype == 1:
                    audio = self.augment_wav.reverberate(audio)
                elif augtype == 2:
                    audio = self.augment_wav.additive_noise('music', audio)
                elif augtype == 3:
                    audio = self.augment_wav.additive_noise('speech', audio)
                elif augtype == 4:
                    audio = self.augment_wav.additive_noise('noise', audio)
            feat.append(audio);
        feat = np.concatenate(feat, axis=0)
        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class Dev_Dataset(Dataset):
    def __init__(self, data_list_path, eval_frames, num_eval, **kwargs):
        self.data_list_path = data_list_path
        df = pd.read_csv(data_list_path)
        self.data_label = df["utt_spk_int_labels"].values
        self.data_list = df["utt_paths"].values
        print("Train Dataset load {} speakers".format(len(np.unique(self.data_label))))
        print("Train Dataset load {} utterance".format(len(self.data_list)))

        self.max_frames = eval_frames
        self.num_eval = num_eval

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True)
        return torch.FloatTensor(audio), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class Test_Dataset(Dataset):
    def __init__(self, data_list, eval_frames, num_eval=10, **kwargs):
        # load data list
        self.data_list = data_list
        self.max_frames = eval_frames
        self.num_eval   = num_eval

    def __getitem__(self, index):
        audio = loadWAV(self.data_list[index], self.max_frames, evalmode=True, num_eval=self.num_eval)
        return torch.FloatTensor(audio), self.data_list[index]

    def __len__(self):
        return len(self.data_list)


class Train_Sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size):
        self.data_source = data_source
        self.label_dict = data_source.label_dict
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size

    def __iter__(self):

        dictkeys = list(self.label_dict.keys());
        dictkeys.sort()

        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        ## Data for each class
        for findex, key in enumerate(dictkeys):
            data = self.label_dict[key]
            numSeg = round_down(min(len(data),self.max_seg_per_spk),self.nPerSpeaker)

            rp = lol(np.random.permutation(len(data))[:numSeg],self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        ## Data in random order
        mixid = np.random.permutation(len(flattened_label))
        mixlabel = []
        mixmap = []

        ## Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        return iter([flattened_list[i] for i in mixmap])

    def __len__(self):
        return len(self.data_source)

if __name__ == "__main__":
    data = loadWAV("test.wav", 100, evalmode=True)
    print(data.shape)
    data = loadWAV("test.wav", 100, evalmode=False)
    print(data.shape)

    def plt_wav(data, name):
        import matplotlib.pyplot as plt
        x = [ i for i in range(len(data[0])) ]
        plt.plot(x, data[0])
        plt.savefig(name)
        plt.close()

    plt_wav(data, "raw.png")
    
    aug_tool = AugmentWAV("data/musan_list.csv", "data/rirs_list.csv", 100)

    audio = aug_tool.reverberate(data)
    plt_wav(audio, "reverb.png")

    audio = aug_tool.additive_noise('music', data)
    plt_wav(audio, "music.png")

    audio = aug_tool.additive_noise('music', data)
    plt_wav(audio, "speech.png")

    audio = aug_tool.additive_noise('music', data)
    plt_wav(audio, "noise.png")

