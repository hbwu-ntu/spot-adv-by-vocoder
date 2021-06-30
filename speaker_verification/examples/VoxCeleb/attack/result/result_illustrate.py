
from argparse import ArgumentParser
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve, roc_auc_score

def get_detection_rate(clean_distance, adv_distance, false_detection_rate):
    clean_distance = np.sort(clean_distance)
    threshold = clean_distance[int(len(clean_distance) * (1 - false_detection_rate))]
    detection_rate = np.sum(adv_distance >= threshold) / len(adv_distance)
    return detection_rate, threshold

def plot_score_hist(num_slots, genuine_distance, adv_distance, figname="Histgram Plot"):
    min_score = min(np.amin(genuine_distance), np.amin(adv_distance))
    max_score = max(np.amax(genuine_distance), np.amax(adv_distance))
    slot_length = (max_score - min_score) / num_slots

    genuine_slots = np.zeros(num_slots)
    for score in genuine_distance:
        index = min(int((score - min_score) / slot_length), num_slots - 1)
        genuine_slots[index] += 1

    adv_slots = np.zeros(num_slots)
    for score in adv_distance:
        index = min(int((score - min_score) / slot_length), num_slots - 1)
        if(index <= 0):
            index = 0
        adv_slots[index] += 1

    centr_scores = np.array([min_score + i * slot_length + slot_length / 2 for i in range(num_slots)], dtype=float)
    x = centr_scores
    plt.figure()
    plt.plot(x, genuine_slots, 'cx--', label='genuine samples')
    plt.plot(x, adv_slots, 'r,--', label='adversarial samples')
    plt.title("Histogram Plot")
    plt.xlabel('Distance')
    plt.ylabel('Amount')
    plt.legend()
    plt.show()
    plt.savefig('./%s.pdf' %(figname), bbox_inches='tight')

def get_far_frr(adv_distance, clean_distance):
    one = np.ones(len(adv_distance))
    zero = np.zeros(len(clean_distance))
    label = np.concatenate((one, zero))
    pre = np.concatenate((adv_distance, clean_distance))
    auc = roc_auc_score(label, pre)
    fpr, tpr, _ = roc_curve(label, pre)
    return fpr, tpr, auc

def roc_plot(vocoder_fpr, vocoder_tpr, lin_fpr, lin_tpr, mel_fpr, mel_tpr, figname="Roc Plot"):
    plt.figure()
    plt.plot(vocoder_fpr, vocoder_tpr, label='Vocoder')
    plt.plot(lin_fpr, lin_tpr, label='GL-lin')
    plt.plot(mel_fpr, mel_tpr, label='GL-mel')
    plt.title("ROC Curve")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()
    plt.savefig('./%s.pdf' %(figname), bbox_inches='tight')

if __name__ == "__main__":
    # args
    parser = ArgumentParser()
    parser.add_argument('--false_detection_rate', help='', type=float, default=0.01)
    parser.add_argument('--num_slots', help='', type=int, default=100)
    parser.add_argument('--clean_score_file', help='', type=str, default="clean/clean_score.lst")
    parser.add_argument('--clean_score_vocoder_file', help='', type=str, default="clean/vocoder_clean_score.lst")
    parser.add_argument('--clean_score_gri_lin_file', help='', type=str, default="clean/clean_griffin_lin_score.lst")
    parser.add_argument('--clean_score_gri_mel_file', help='', type=str, default="clean/clean_griffin_mel_score.lst")
    parser.add_argument('--adv_score_file', help='', type=str, default="ep15/adv_trials_score.lst")
    parser.add_argument('--adv_score_vocoder_file', help='', type=str, default="ep15/vocoder_adv_trials_score.lst")
    parser.add_argument('--adv_score_gri_lin_file', help='', type=str, default="ep15/adv_griffin_lin_trials_score.lst")
    parser.add_argument('--adv_score_gri_mel_file', help='', type=str, default="ep15/adv_griffin_mel_trials_score.lst")
    args = parser.parse_args()

    false_detection_rate = args.false_detection_rate
    num_slots = args.num_slots
    adv_file = args.adv_score_file
    adv_vocoder_file = args.adv_score_vocoder_file
    adv_lin_file = args.adv_score_gri_lin_file
    adv_mel_file = args.adv_score_gri_mel_file
    clean_file = args.clean_score_file
    clean_vocoder_file = args.clean_score_vocoder_file
    clean_mel_file = args.clean_score_gri_mel_file
    clean_lin_file = args.clean_score_gri_lin_file

    adv_scores = np.array(np.loadtxt(adv_file, dtype=str).T[3], dtype=np.float64)
    adv_vocoder_scores = np.array(np.loadtxt(adv_vocoder_file, dtype=str).T[3], dtype=np.float64)
    adv_lin_scores = np.array(np.loadtxt(adv_lin_file, dtype=str).T[3], dtype=np.float64)
    adv_mel_scores = np.array(np.loadtxt(adv_mel_file, dtype=str).T[3], dtype=np.float64)
    clean_scores = np.array(np.loadtxt(clean_file, dtype=str).T[3], dtype=np.float64)
    clean_vocoder_scores = np.array(np.loadtxt(clean_vocoder_file, dtype=str).T[3], dtype=np.float64)
    clean_lin_scores = np.array(np.loadtxt(clean_lin_file, dtype=str).T[3], dtype=np.float64)
    clean_mel_scores = np.array(np.loadtxt(clean_mel_file, dtype=str).T[3], dtype=np.float64)

    clean_distance = np.absolute(clean_scores - clean_vocoder_scores)
    adv_distance = np.absolute(adv_vocoder_scores - adv_scores)
    clean_lin_distance = np.absolute(clean_lin_scores - clean_scores)
    clean_mel_distance = np.absolute(clean_mel_scores - clean_scores)
    adv_lin_distance = np.absolute(adv_lin_scores - adv_scores)
    adv_mel_distance = np.absolute(adv_mel_scores - adv_scores)

    # get detection rate of adversarial samples under the false detection rate of genuine sampels
    vocoder_detection_rate, threshold = get_detection_rate(clean_distance, adv_distance, false_detection_rate)
    mel_detection_rate, threshold = get_detection_rate(clean_mel_distance, adv_mel_distance, false_detection_rate)
    lin_detection_rate, threshold = get_detection_rate(clean_lin_distance, adv_lin_distance, false_detection_rate)
    print("When false detection rate is: {}".format(false_detection_rate))
    print("Vocoder detection rate is: {}".format(vocoder_detection_rate))
    print("Griffin-Lim linear detection rate is: {}".format(lin_detection_rate))
    print("Griffin-Lim mel detection rate is: {} \n".format(mel_detection_rate))

    # plot histogram
    hist_figname = "result/Histgram Plot"
    plot_score_hist(num_slots, clean_distance, adv_distance, hist_figname)

    # plot roc curve, return the auc(the area under the curve)  
    vocoder_fpr, vocoder_tpr, vocoder_auc = get_far_frr(adv_distance, clean_distance)
    lin_fpr, lin_tpr, lin_auc = get_far_frr(adv_lin_distance, clean_lin_distance)
    mel_fpr, mel_tpr, mel_auc = get_far_frr(adv_mel_distance, clean_mel_distance)
    print("Vocoder auc:", vocoder_auc)
    print("Griffin-Lim mel auc:", mel_auc)
    print("Griffin-Lim lin auc:", lin_auc)
    roc_figname = "result/Roc Plot"
    roc_plot(vocoder_fpr, vocoder_tpr, lin_fpr, lin_tpr, mel_fpr, mel_tpr, figname=roc_figname)
