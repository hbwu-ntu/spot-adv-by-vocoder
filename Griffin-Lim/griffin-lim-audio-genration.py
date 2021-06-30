from argparse import ArgumentParser
import os
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from audio import AudioSpec
anal = AudioSpec()

def recon_lin(wav_path):
    src, tgt = wav_path[0], wav_path[1]
    x = anal.load_wav(src)
    lin = anal.lin(x)
    inv_lin = anal.inv_lin(lin)
    anal.save_wav(inv_lin, tgt)

def recon_mel(wav_path):
    src, tgt = wav_path[0], wav_path[1]
    x = anal.load_wav(src)
    lin = anal.lin(x)
    mel = anal.mel(lin)
    inv_mel = anal.inv_mel(mel)
    anal.save_wav(inv_mel, tgt)

if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument('--ori_audio_dir', help='', type=str, default="/work/jason410/speaker_code/examples/VoxCeleb/attack/data/adv_data/wav")
    parser.add_argument('--save_dir', help='', type=str, default="exp/clean")
    parser.add_argument('--is_linear_spectrogram', action='store_true', default=False)
    parser.add_argument('--num_jobs', help='', type=int, default=64)
    args = parser.parse_args()
    
    ori_audio_dir = args.ori_audio_dir
    save_dir = args.save_dir
    is_linear_spectrogram = args.is_linear_spectrogram
    num_jobs = args.num_jobs
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
	
    wav_pth = []
    for x in os.listdir(ori_audio_dir):
        wav_pth.append([
            os.path.join(ori_audio_dir, x),
            os.path.join(save_dir, x)])
    
    with Pool(args.num_jobs) as p:
        if is_linear_spectrogram: 
            p.map(recon_lin, wav_pth)
        else:
            p.map(recon_mel, wav_pth)
