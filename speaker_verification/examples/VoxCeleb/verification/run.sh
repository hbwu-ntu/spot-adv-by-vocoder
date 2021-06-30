#!/bin/bash

voxceleb1_path=/mnt/ssd-201-112-01/cpii.local/bzheng/voxceleb1
voxceleb2_path=/mnt/ssd-201-112-01/cpii.local/hbwu/voxceleb2
musan_path=/mnt/ssd-201-112-01/cpii.local/hbwu/musan
rirs_path=/mnt/ssd-201-112-01/cpii.local/hbwu/RIRS_NOISES
trials_path=data/trials.lst

nnet_type=ResNet34_quarter
loss_type=amsoftmax

vad=false
embedding_dim=256

. ./path.sh

stage=2
echo stage $stage

# build soft link
if [ $stage -eq 0 ];then
	if [ ! -d data/wav_files ]; then
		mkdir -p data/wav_files
	fi

	rm -rf data/wav_files/dev
	mkdir -p data/wav_files/dev
	# format voxceleb1
	ln -s ${voxceleb1_path}/dev/wav/id* data/wav_files/dev/

	# format voxceleb2
	ln -s ${voxceleb2_path}/dev/aac/id* data/wav_files/dev/

	# wget https://www.openslr.org/resources/49/voxceleb1_test_v2.txt
	cp voxceleb1_test_v2.txt data/voxceleb1_test_v2.txt
fi


# build data list
if [ $stage -eq 1 ];then
	extension=wav

	echo build dev data list
	python3 $SPEAKER_TRAINER_ROOT/scripts/build_datalist.py \
		--extension $extension \
		--dataset_dir data/wav_files/dev \
		--data_list_path data/dev_list.csv

	echo build musan data list
	python3 $SPEAKER_TRAINER_ROOT/scripts/build_datalist.py \
		--extension wav \
		--dataset_dir $musan_path \
		--data_list_path data/musan_list.csv

	echo build rirs data list
	python3 $SPEAKER_TRAINER_ROOT/scripts/build_datalist.py \
		--extension wav \
		--dataset_dir $rirs_path \
		--data_list_path data/rirs_list.csv

	rm -rf $trials_path
	python3 local/format_trials.py \
			--voxceleb1_root $voxceleb1_path \
			--src_trials_path data/voxceleb1_test_v2.txt \
			--dst_trials_path $trials_path
fi

# train
if [ $stage -eq 2 ];then
	CUDA_VISIBLE_DEVICES=1 python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
		--nnet_type $nnet_type \
		--loss_type $loss_type \
		--batch_size 128 \
		--num_workers 32 \
		--embedding_dim $embedding_dim \
		--save_top_k 50 \
		--train_list_path data/dev_list.csv \
		--musan_list_path data/musan_list.csv \
		--rirs_list_path data/rirs_list.csv \
		--max_epochs 50 \
		--max_frames 201 --min_frames 200 \
		--learning_rate 0.01 \
		--lr_step_size 10 \
		--distributed_backend dp \
		--trials_path $trials_path \
		--eval_interval -1 \
		--nPerSpeaker 1 \
		--reload_dataloaders_every_epoch \
		--gpus 1
fi
# evaluate
if [ $stage -eq 3 ];then
	ckpt_path=/users/cpii.local/hbwu/adv_detect/speaker_verification/pretrained_model/ckpt.pt
	CUDA_VISIBLE_DEVICES=0 python3 -W ignore $SPEAKER_TRAINER_ROOT/main.py \
		--batch_size 64 \
		--nnet_type $nnet_type \
		--num_workers 32 \
		--train_list_path data/dev_list.csv \
		--trials_path data/trials.lst \
		--gpus 1 \
		--checkpoint_path $ckpt_path \
		--evaluate

	rm -rf lightning_logs
fi

