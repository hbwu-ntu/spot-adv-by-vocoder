score=score.txt # every line: enroll_file test_file score
trial=trial.txt # every line: enroll_file test_file target/nontarget

thresh=-99999 # for calculate eer
# thresh=0 # for calculate far, frr
python2 compute_asv_performance_python.py --trials $trial --scores $score --threshold $thresh