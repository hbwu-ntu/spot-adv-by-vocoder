false_detection_rate=0.01
num_slots=100
clean_score_file=result/trials_score.lst
clean_score_vocoder_file=result/trials_vocoder_score.lst
clean_score_gri_lin_file=result/trials_gri_lin_score.lst
clean_score_gri_mel_file=result/trials_gri_mel_score.lst
adv_score_file=result/adv_trials_score.lst
adv_score_vocoder_file=result/adv_trials_vocoder_score.lst
adv_score_gri_lin_file=result/adv_trials_gri_lin_score.lst
adv_score_gri_mel_file=result/adv_trials_gri_mel_score.lst

python3 result/result_illustrate.py \
        --false_detection_rate $false_detection_rate \
        --num_slots $num_slots \
        --clean_score_file $clean_score_file \
        --clean_score_vocoder_file $clean_score_vocoder_file \
        --clean_score_gri_lin_file $clean_score_gri_lin_file \
        --clean_score_gri_mel_file $clean_score_gri_mel_file \
        --adv_score_file $adv_score_file \
        --adv_score_vocoder_file $adv_score_vocoder_file \
        --adv_score_gri_lin_file $adv_score_gri_lin_file \
        --adv_score_gri_mel_file $adv_score_gri_mel_file