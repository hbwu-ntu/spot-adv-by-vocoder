
# data_root=../speaker_verification/examples/VoxCeleb/attack/data/adv_data_epsilon15_it5
data_root=../speaker_verification/examples/VoxCeleb/attack/data/clean
ori_audio_dir=$data_root/wav
save_dir=$data_root/gri_lin
num_jobs=32

python3 -B griffin-lim-audio-genration.py \
    --ori_audio_dir $ori_audio_dir \
    --save_dir $save_dir \
    --num_jobs $num_jobs \
    --is_linear_spectrogram

save_dir=$data_root/gri_mel
python3 -B griffin-lim-audio-genration.py \
    --ori_audio_dir $ori_audio_dir \
    --save_dir $save_dir \
    --num_jobs $num_jobs
