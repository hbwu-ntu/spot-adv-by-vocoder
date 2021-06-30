#!/usr/bin/env python
# encoding: utf-8

from tqdm import tqdm
import numpy as np
from .compute_eer import compute_eer

def length_norm(vector):
    assert len(vector.shape) == 1
    dim = len(vector)
    norm = np.linalg.norm(vector)
    return np.sqrt(dim) * vector / norm

def cosine_score(trials, index_mapping, eval_vectors, apply_length_norm=False):
    target_scores = []
    nontarget_scores = []
    if apply_length_norm:
        print("apply length norm")
    for item in trials:
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        if apply_length_norm:
            enroll_vector = length_norm(enroll_vector)
            test_vector = length_norm(test_vector)
        score = enroll_vector.dot(test_vector.T)
        denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
        score = score/denom
        if item[0] == "1":
            target_scores.append(score)
        else:
            nontarget_scores.append(score)
    eer, th = compute_eer(target_scores, nontarget_scores)
    return eer, th

def save_cosine_score(trials, index_mapping, eval_vectors, score_file, apply_length_norm=False):
    target_scores = []
    nontarget_scores = []
    if apply_length_norm:
        print("apply length norm")
    with open(score_file, "w") as f:
        for item in trials:
            enroll_vector = eval_vectors[index_mapping[item[1]]]
            test_vector = eval_vectors[index_mapping[item[2]]]
            if apply_length_norm:
                enroll_vector = length_norm(enroll_vector)
                test_vector = length_norm(test_vector)
            score = enroll_vector.dot(test_vector.T)
            denom = np.linalg.norm(enroll_vector) * np.linalg.norm(test_vector)
            score = score/denom
            f.write("{} {} {} {}\n".format(item[0], item[1], item[2], score))
            if item[0] == "1":
                target_scores.append(score)
            else:
                nontarget_scores.append(score)
    eer, th = compute_eer(target_scores, nontarget_scores)
    return eer, th

def PLDA_score(trials, index_mapping, eval_vectors, plda_analyzer):
    target_scores = []
    nontarget_scores= []
    for item in trials:
        enroll_vector = eval_vectors[index_mapping[item[1]]]
        test_vector = eval_vectors[index_mapping[item[2]]]
        score = plda_analyzer.NLScore(enroll_vector, test_vector)
        if item[0] == "1":
            target_scores.append(score)
        else:
            nontarget_scores.append(score)
    eer, th = compute_eer(target_scores, nontarget_scores)
    return eer, th



