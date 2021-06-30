from argparse import ArgumentParser
import os

import torch
from scipy.io import wavfile
import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from speaker_trainer import Model
from pytorch_lightning.callbacks import Callback

from speaker_trainer.backend import compute_eer
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')

class Adversarial_Attack_Helper(object):
    def __init__(self, model, alpha=3.0, restarts=1, num_iters=5, epsilon=15, adv_save_dir="data/adv_data", device="cuda"):
        self.model = model
        self.alpha = alpha
        self.restarts = restarts
        self.num_iters = num_iters
        self.epsilon = epsilon
        adv_save_dir = adv_save_dir + '_epsilon{}_it{}'.format(epsilon, num_iters)
        self.adv_save_dir = adv_save_dir

        if not os.path.exists(os.path.join(adv_save_dir, "wav")):
            os.makedirs(os.path.join(adv_save_dir, "wav"))

        self.trials = self.model.trials
        self.adv_trials_path = os.path.join(adv_save_dir, "adv_trials.lst")
        self.device = device
        self.model.eval()
        if self.device == "cuda":
            self.model.cuda()

    def evaluate(self, trials=None):
        if trials is None:
            trials = self.trials
        with torch.no_grad():
            eer, th = self.model.cosine_evaluate(trials) 

    def attack(self):
        # adversarial attack example generation
        if os.path.exists(self.adv_trials_path):
            os.remove(self.adv_trials_path)
        adv_trials_file = open(self.adv_trials_path, "a+")
        target_score = []
        nontarget_score = []
        for idx, item in enumerate(tqdm(self.trials)):
            label, enroll_path, adv_test_path, score = self.pgd_adversarial_attack_step(idx, item)
            adv_trials_file.write("{} {} {}\n".format(label, enroll_path, adv_test_path))
            if label == 0:
                nontarget_score.append(score)
            else:
                target_score.append(score)
        eer, th = compute_eer(target_score, nontarget_score)
        print("EER: {:.3f} %".format(eer*100))
        return self.adv_trials_path

    def pgd_adversarial_attack_step(self, idx, item):
        label = item[0]
        enroll_path = item[1]
        test_path = item[2]

        # load data
        samplerate, enroll_wav = self.load_wav(enroll_path)
        samplerate, test_wav = self.load_wav(test_path)
        max_delta = torch.zeros_like(test_wav).cuda()

        # init best_score and alpha
        label = int(label)
        if label == 1:
            best_score = torch.tensor(float('inf')).cuda()
            alpha = self.alpha*(-1.0)
        else:
            best_score = torch.tensor(float('-inf')).cuda()
            alpha = self.alpha*(1.0)

        enroll_embedding = self.model.extract_speaker_embedding(enroll_wav).squeeze(0)
        for i in range(self.restarts):
            delta = torch.zeros_like(test_wav, requires_grad=True).cuda()
            for t in range(self.num_iters):
                # extract test speaker embedding
                test_embedding = self.model.extract_speaker_embedding(test_wav + delta).squeeze(0)
                # cosine score
                score = enroll_embedding.dot(test_embedding.T)
                denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
                score = score/denom

                # compute grad and update delta
                score.backward(retain_graph=True)
                delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-1*self.epsilon, self.epsilon)
                delta.grad.zero_()

            test_embedding = self.model.extract_speaker_embedding(test_wav+delta).squeeze(0)
            final_score = enroll_embedding.dot(test_embedding.T)
            denom = torch.norm(enroll_embedding) * torch.norm(test_embedding)
            final_score = final_score/denom

            if label == 1 and best_score >= final_score:
                max_delta = delta.data
                best_score = torch.min(best_score, final_score)
            elif label == 0 and best_score <= final_score:
                max_delta = delta.data
                best_score = torch.max(best_score, final_score)

        # Get Adversarial Attack wav
        adv_wav = test_wav + max_delta
        adv_wav = adv_wav.cpu().detach().numpy()
        final_score = final_score.cpu().detach().numpy()

        # save attack test wav
        idx = '%08d' % idx
        adv_test_path = os.path.join(self.adv_save_dir, "wav", idx+".wav")
        wavfile.write(adv_test_path, samplerate, adv_wav.astype(np.int16))
        return label, enroll_path, adv_test_path, final_score

    def load_wav(self, path):
        sample_rate, audio = wavfile.read(path)
        audio = torch.FloatTensor(audio)
        if self.device == "cuda":
            audio = audio.cuda()
        return sample_rate, audio


if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument('--alpha', help='', type=float, default=3.0)
    parser.add_argument('--restarts', help='', type=int, default=1)
    parser.add_argument('--num_iters', help='', type=int, default=5)
    parser.add_argument('--epsilon', help='', type=int, default=15)
    parser.add_argument('--adv_save_dir', help='', type=str, default="data/adv_data")
    parser.add_argument('--device', help='', type=str, default="cuda")
    parser.add_argument('--evaluate_only', action='store_true', default=False)
    parser = Trainer.add_argparse_args(parser)
    parser = Model.add_model_specific_args(parser)
    args = parser.parse_args()

    model = Model(**vars(args))

    # 0. pop loss Function parameter
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")["state_dict"]
    loss_weights = []
    for key, value in state_dict.items():
        if "loss" in key:
            loss_weights.append(key)
    for item in loss_weights:
        state_dict.pop(item)

    # load speaker encoder state dict and init the attack helper
    model.load_state_dict(state_dict, strict=False)
    print("initial parameter from pretrain model {}".format(args.checkpoint_path))
    helper = Adversarial_Attack_Helper(model, args.alpha, args.restarts, args.num_iters, args.epsilon, args.adv_save_dir, args.device)

    if (args.evaluate_only):
        print("evaluate in trials {}".format(args.trials_path))
        helper.evaluate()
    else:
        print("evaluate in raw data")
        helper.evaluate()
        print("attacking ")
        helper.attack()
