# Spotting adversarial samples for ASV by neural vocoders

## Installation
```bash
git clone https://github.com/HaibinWu666/spot-adv-by-vocoder.git
pip install -r requirements.txt
cd ParallelWaveGAN
pip install -e .
```

## Usage
### 1.Train ASV 
You can skip this step and use our pretrained model (speaker_verification/pretrained_model/ckpt.pt)
- prepare the data
```bash
cd speaker_verification/examples/VoxCeleb/verification
set the voxceleb1_path、voxceleb2_path、musan_path、rirs_path in run.sh
set the stage in run.sh to 0 to build soft links
bash run.sh
set the stage in run.sh to 1 to format data
bash run.sh
```


## Citation

## Reference
- The implementation of ParallelWaveGAN is from https://github.com/kan-bayashi/ParallelWaveGAN
