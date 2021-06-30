##
import argparse
import sys

def compute_eer(trials, scores):

	trial_scores = {}
	with open(scores, 'r') as f:
		for line in f.readlines():
			enrollid, testid, score = line.split()
			trial_scores.update({enrollid+'_'+testid:float(score)})

	# print('gt_A is %d, gt_R is %d.' %(gt_A, gt_R))
	trial_ids = trial_scores.keys()
    # trial_ids.sorted(key=lambda x:trial_scores.get(x), reverse=True)
	trial_ids.sort(key=lambda x:trial_scores.get(x), reverse=True)
	# print(trial_ids[0])

	trial_gt = {}
	gt_A = 0
	gt_R = 0
	with open(trials, 'r') as f:
		for line in f.readlines():
			enrollid, testid, gt = line.split()
			trialid = enrollid+'_'+testid
			if trial_scores.get(trialid) != None:
				trial_gt.update({trialid:gt})
				if gt == 'target':
					gt_A += 1
				else:
					gt_R += 1


	threshold_index = -1
	FAR = 0.0
	FRR = 1.0
	FA = 0
	FR = gt_A

	while FRR > FAR:
		threshold_index += 1
		if trial_gt.get(trial_ids[threshold_index]) == 'target':
			FR -= 1
		else:
			FA += 1
		FAR = float(FA)/gt_R
		FRR = float(FR)/gt_A

	threshold = trial_scores.get(trial_ids[threshold_index])
	# print(threshold_index)
	EER = (FAR+FRR)/2

	return threshold, EER

def compute_far(trials, scores, threshold):

	trial_scores = {}
	with open(scores, 'r') as f:
		for line in f.readlines():
			enrollid, testid, score = line.split()
			trial_scores.update({enrollid+'_'+testid:float(score)})

	trial_ids = trial_scores.keys()
	trial_ids.sort(key=lambda x:trial_scores.get(x), reverse=True)

	trial_gt = {}
	gt_A = 0
	gt_R = 0
	with open(trials, 'r') as f:
		for line in f.readlines():
			enrollid, testid, gt = line.split()
			trialid = enrollid+'_'+testid
			if trial_scores.get(trialid) != None:
				trial_gt.update({trialid:gt})
				if gt == 'target':
					gt_A += 1
				else:
					gt_R += 1

	threshold_index = 0
	FA = 0

	while trial_scores.get(trial_ids[threshold_index]) >= threshold:
		if trial_gt.get(trial_ids[threshold_index]) == 'nontarget':
			FA += 1
		threshold_index += 1
		if threshold_index >= len(trial_ids):
			break

	FAR = float(FA)/gt_R

	return FAR

def compute_frr(trials, scores, threshold):

	trial_scores = {}
	with open(scores, 'r') as f:
		for line in f.readlines():
			enrollid, testid, score = line.split()
			trial_scores.update({enrollid+'_'+testid:float(score)})

	trial_ids = trial_scores.keys()
	trial_ids.sort(key=lambda x:trial_scores.get(x))

	trial_gt = {}
	gt_A = 0
	gt_R = 0
	with open(trials, 'r') as f:
		for line in f.readlines():
			enrollid, testid, gt = line.split()
			trialid = enrollid+'_'+testid
			if trial_scores.get(trialid) != None:
				trial_gt.update({trialid:gt})
				if gt == 'target':
					gt_A += 1
				else:
					gt_R += 1

	threshold_index = 0
	FR = 0

	while trial_scores.get(trial_ids[threshold_index]) < threshold:
		if trial_gt.get(trial_ids[threshold_index]) == 'target':
			FR += 1
		threshold_index += 1
		if threshold_index >= len(trial_ids):
			break

	FRR = float(FR)/gt_A

	return FRR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Metrics of SV systems.')
    parser.add_argument('--trials', 
        default="data/voxceleb1_test/trials_adv", type=str)
    parser.add_argument('--scores', default="exp/score_adv_stft_sigma5.0", type=str)
    parser.add_argument('--threshold', default=-99999, type=float)

    args = parser.parse_args()

    if args.threshold == -99999:
    	threshold, EER = compute_eer(args.trials, args.scores)
    	print('The threshold is %f, EER is %f.' %(threshold, EER))
    else:
    	FAR = compute_far(args.trials, args.scores, args.threshold)
    	FRR = compute_frr(args.trials, args.scores, args.threshold)
    	print('The FAR is %f, FRR is %f, under the threshold of %f.' %(FAR, FRR, args.threshold))

