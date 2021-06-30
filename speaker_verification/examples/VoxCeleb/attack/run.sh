#!/bin/bash

voxceleb1_path=/users/cpii.local/hbwu/voxceleb1
trials_path=data/trials.lst

nnet_type=ResNet34_quarter
loss_type=amsoftmax

# ckpt_path=/users/cpii.local/hbwu/adv_detect/speaker_verification/pretrained_model/ckpt.pt
ckpt_path=/users/cpii.local/hbwu/adv_detect/speaker_verification/examples/VoxCeleb/verification/exp/ResNet34_quarter_ASP_amsoftmax_2021-06-27-15-02-18/epoch=45_train_loss=1.30.ckpt
. ./path.sh

stage=1
echo stage $stage

# format data dir structure by soft link
if [ $stage -eq 0 ];then
    if [ ! -d data ]; then
        mkdir -p data
    fi
    rm -rf $trials_path

    # wget https://www.openslr.org/resources/49/voxceleb1_test_v2.txt
    cp voxceleb1_test_v2.txt data/voxceleb1_test_v2.txt
    python3 local/format_trials.py \
	    --voxceleb1_root $voxceleb1_path \
	    --src_trials_path data/voxceleb1_test_v2.txt \
	    --dst_trials_path $trials_path
fi

# attack
if [ $stage -eq 1 ];then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore $SPEAKER_TRAINER_ROOT/adversarial_attack.py \
	    --checkpoint_path $ckpt_path \
		--nnet_type $nnet_type \
	    --alpha 2 \
	    --restarts 1 \
	    --num_iters 5 \
	    --epsilon 10 \
	    --device cuda \
	    --trials_path $trials_path \
		--evaluate 
   rm -rf lightning_logs
fi

# evaluate adversarial performance
data_root=adv_data_epsilon15_it5
trial_file=adv_trials.lst
adv_trial=data/$data_root/$trial_file
if [ $stage -eq 2 ];then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore $SPEAKER_TRAINER_ROOT/adversarial_attack.py \
	    --checkpoint_path $ckpt_path \
		--nnet_type $nnet_type \
	    --device cuda \
	    --trials_path $adv_trial \
		--evaluate \
		--evaluate_only
   rm -rf lightning_logs
   cp $adv_trial data/adv_trials.lst
   rm data/$data_root/adv_trials_score.lst
fi

# prepare trials
if [ $stage -eq 3 ];then
    CUDA_VISIBLE_DEVICES=0 python3 -W ignore local/prepare_trials.py \
	    --input_trial_file data/trials.lst \
		--output_trial_file data/clean/trials.lst \
		--output_wav_dir data/clean/wav
	CUDA_VISIBLE_DEVICES=0 python3 -W ignore local/prepare_trials.py \
	    --input_trial_file data/adv_trials.lst \
		--output_trial_file $adv_trial \
		--output_wav_dir data/adv_data_epsilon15_it5/wav \
		--is_adv
fi

# evaluate and save score files
if [ $stage -eq 4 ];then
    for adv_trial in `find data/adv_data_epsilon15_it5/ -name "*.lst"`; 
    do
        CUDA_VISIBLE_DEVICES=0 python3 -W ignore $SPEAKER_TRAINER_ROOT/adversarial_attack.py \
            --checkpoint_path $ckpt_path \
            --nnet_type $nnet_type \
            --device cuda \
            --trials_path $adv_trial \
            --evaluate \
            --evaluate_only;
    done
	for gen_trial in `find data/clean/ -name "*.lst"`; 
    do
        CUDA_VISIBLE_DEVICES=0 python3 -W ignore $SPEAKER_TRAINER_ROOT/adversarial_attack.py \
            --checkpoint_path $ckpt_path \
            --nnet_type $nnet_type \
            --device cuda \
            --trials_path $gen_trial \
            --evaluate \
            --evaluate_only;
    done
fi

# result illustration
if [ $stage -eq 5 ];then
	cp data/adv_data_epsilon15_it5/*score.lst result/
	cp data/clean/*score.lst result/
	bash ./result/result_illustrate.sh
fi
