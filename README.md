# Spotting adversarial samples for ASV by neural vocoders

## Introduction
- This repo is used for reproducing the main result of "ADVERSARIAL SAMPLE DETECTION FOR SPEAKER VERIFICATION BY NEURAL VOCODERS"
- Paper: [Arxiv]
- Audio demo: [Demo](https://haibinwu666.github.io/adv-audio-demo/index.html)

## Installation
```bash
git clone https://github.com/HaibinWu666/spot-adv-by-vocoder.git
pip install -r requirements.txt
cd ParallelWaveGAN
pip install -e .
```

## Usage
### 1.Train ASV 
- Prepare the data
```bash
cd speaker_verification/examples/VoxCeleb/verification
set the voxceleb1_path、voxceleb2_path、musan_path、rirs_path in run.sh (voxceleb1_path and voxceleb2_path should be formated as voxceleb1_path/dev/wav/idxxx and voxceleb2_path/dev/aac/idxxx;)
set the stage in run.sh to 0 to build soft links
bash run.sh
set the stage in run.sh to 1 to format data
bash run.sh
```
- Train 

**You can skip this step and use our pretrained model (speaker_verification/pretrained_model/)**
```bash
set the stage in run.sh to 2
bash run.sh
the model is saved in exp/
```
- Evaluate
```bash
set the stage in run.sh to 3
set the ckpt_path
bash run.sh
```

### 2. Generate adversarial samples
- Prepare data
```bash
set the stage in run.sh to 0
set the voxceleb1_path
bash run.sh
```
- Attack
```bash
set the stage in run.sh to 1
set the voxceleb1_path
bash run.sh
```
- Evaluate
```bash
set the stage in run.sh to 2
bash run.sh
```
- Prepare trial files
```bash
set the stage in run.sh to 3
bash run.sh
```

### 3. Train vocoder
**You can skip this step and use our pretrained model in ParallelWaveGAN/pretrained_model/**
- Prepare the data and train
```bash
cd ParallelWaveGAN/egs/ljspeech/voxceleb1
set the voxceleb1_path in run.sh
set the stage in run.sh to 2
bash run.sh
```

### 4. Resynthesis the wav
- Use vocoder to resynthesis the wav
```bash
For adversarial audio
cd ParallelWaveGAN
set model_dir and data_dir in run_audio_generation.sh
set model_dir=pretrained_model/train_nodev_ljspeech_parallel_wavegan.v1.long 
set data_dir=../speaker_verification/examples/VoxCeleb/attack/data/adv_data_epsilon15_it5
bash run_audio_generation.sh
For clean audio
cd ParallelWaveGAN
set model_dir=pretrained_model/train_nodev_ljspeech_parallel_wavegan.v1.long
set data_dir=../speaker_verification/examples/VoxCeleb/attack/data/clean
bash run_audio_generation.sh
```
- Use Griffin-Lim to resynthesis the wav
```bash
For adversarial audio
cd Griffin-Lim
set data_root in run.sh=../speaker_verification/examples/VoxCeleb/attack/data/adv_data_epsilon15_it5
bash run.sh
For clean audio
cd Griffin-Lim
set data_root in run.sh=../speaker_verification/examples/VoxCeleb/attack/data/clean
bash run.sh
```

### 5. Result illustration
```bash
cd speaker_verification/examples/VoxCeleb/attack
set stage to 5 in run.sh
bash run.sh
```

## Citation
If you think this work helps your research or use the code, please consider citing our paper. Thank you!

## Reference
- The implementation of ParallelWaveGAN is from https://github.com/kan-bayashi/ParallelWaveGAN
- https://github.com/thuhcsi/torch_speaker
- https://github.com/clovaai/voxceleb_trainer
