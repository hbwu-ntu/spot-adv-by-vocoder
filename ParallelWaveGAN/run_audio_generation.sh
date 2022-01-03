stage=0
echo stage=${stage}

data_dir=/work/jason410/speaker_code/examples/VoxCeleb/attack/data/adv_data

echo $data_dir
#### convert waveform to melspec ####
if [ $stage -eq 0 ]; then
    parallel-wavegan-preprocess \
        --config pretrained_model/train_nodev_ljspeech_parallel_wavegan.v1.long/config.yml \
        --rootdir $data_dir/wav \
        --dumpdir $data_dir/samples/dump/row
fi

#### normlize melspec ####
if [ $stage -le 1 ]; then
    parallel-wavegan-normalize \
        --config pretrained_model/train_nodev_ljspeech_parallel_wavegan.v1.long/config.yml \
        --rootdir $data_dir/samples/dump/row \
        --dumpdir $data_dir/samples/dump/norm \
        --stats pretrained_model/train_nodev_ljspeech_parallel_wavegan.v1.long/stats.h5
fi

#### generate waveform ####
if [ $stage -le 2 ]; then
    parallel-wavegan-decode \
        --checkpoint pretrained_model/train_nodev_ljspeech_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl \
        --dumpdir $data_dir/samples/dump/norm \
        --outdir $data_dir/generated_wav
fi
