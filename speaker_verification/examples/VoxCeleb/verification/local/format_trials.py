#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import numpy as np
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--voxceleb1_root', help='voxceleb1_root', type=str, default="/ceph/home/zhangy20/datasets/VoxCeleb/voxceleb1/")
    parser.add_argument('--src_trials_path', help='src_trials_path', type=str, default="voxceleb1_test_v2.txt")
    parser.add_argument('--dst_trials_path', help='dst_trials_path', type=str, default="new_trials.lst")
    parser.add_argument('--apply_vad', action='store_true', default=False)
    args = parser.parse_args()

    trials = np.loadtxt(args.src_trials_path, dtype=str)

    f = open(args.dst_trials_path, "a+")
    for item in trials:
        enroll_path = os.path.join(args.voxceleb1_root, "test/wav", item[1])
        test_path = os.path.join(args.voxceleb1_root, "test/wav", item[2])
        if args.apply_vad:
            enroll_path = enroll_path.strip("*.wav") + "*.vad"
            test_path = test_path.strip("*.wav") + "*.vad"
        f.write("{} {} {}\n".format(item[0], enroll_path, test_path))

