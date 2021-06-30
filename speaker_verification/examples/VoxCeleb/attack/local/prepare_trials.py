#! /usr/bin/python
# -*- encoding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from scipy.io import wavfile
from argparse import ArgumentParser
from tqdm import tqdm

def loadWAV(filename):
	sample_rate, audio  = wavfile.read(filename)
	return audio

def generate_trials():
	parser = ArgumentParser()

	parser.add_argument('--input_trial_file', type=str, default="data/trials.lst")
	parser.add_argument('--output_trial_file', type=str, default="data/clean/trials.lst")
	parser.add_argument('--output_wav_dir', type=str, default="data/clean/wav")
	parser.add_argument('--is_adv', action='store_true', default=False)
	args = parser.parse_args()

	if not os.path.exists(args.output_wav_dir):
		os.makedirs(args.output_wav_dir)

	test_wavs = np.unique(np.loadtxt(args.input_trial_file, dtype=str).T[2])

	raw2aug_mapping = {}
	for idx, test_wav in tqdm(enumerate(test_wavs)):
		if (args.is_adv):
			idx = '%08d' % idx
			raw2aug_mapping[test_wav] = idx
		else:
			raw2aug_mapping[test_wav] = idx
			audio = loadWAV(test_wav)
			wavfile.write(os.path.join(args.output_wav_dir, str(idx)+".wav"), 16000, audio.astype(np.int16))

	with open(args.output_trial_file, "w") as f:
		for item in np.loadtxt(args.input_trial_file, dtype=str):
			test_wav_path = os.path.join(args.output_wav_dir, str(raw2aug_mapping[item[2]])+".wav")
			f.write("{} {} {}\n".format(item[0], item[1], test_wav_path))
	
	with open(args.output_trial_file[:-4]+"_vocoder.lst", "w") as f:
		for item in np.loadtxt(args.input_trial_file, dtype=str):
			test_wav_path = os.path.join(args.output_wav_dir[:-3]+"generated_wav", str(raw2aug_mapping[item[2]])+".wav")
			f.write("{} {} {}\n".format(item[0], item[1], test_wav_path))

	with open(args.output_trial_file[:-4]+"_gri_lin.lst", "w") as f:
		for item in np.loadtxt(args.input_trial_file, dtype=str):
			test_wav_path = os.path.join(args.output_wav_dir[:-3]+"gri_lin", str(raw2aug_mapping[item[2]])+".wav")
			f.write("{} {} {}\n".format(item[0], item[1], test_wav_path))
	
	with open(args.output_trial_file[:-4]+"_gri_mel.lst", "w") as f:
		for item in np.loadtxt(args.input_trial_file, dtype=str):
			test_wav_path = os.path.join(args.output_wav_dir[:-3]+"gri_mel", str(raw2aug_mapping[item[2]])+".wav")
			f.write("{} {} {}\n".format(item[0], item[1], test_wav_path))

if __name__ == '__main__':
	generate_trials()