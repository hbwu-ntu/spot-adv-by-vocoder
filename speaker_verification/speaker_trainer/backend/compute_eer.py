
def compute_eer(target_scores, nontarget_scores):
    if isinstance(target_scores , list) is False:
        target_scores = list(target_scores)
    if isinstance(nontarget_scores , list) is False:
        nontarget_scores = list(nontarget_scores)

    target_scores = sorted(target_scores)
    nontarget_scores = sorted(nontarget_scores)
    target_size = len(target_scores);
    nontarget_size = len(nontarget_scores)

    target_position = 0
    for i in range(target_size-1):
        target_position = i
        nontarget_n = nontarget_size * float(target_position) / target_size
        nontarget_position = int(nontarget_size - 1 - nontarget_n)
        if nontarget_position < 0:
            nontarget_position = 0
        if nontarget_scores[nontarget_position] < target_scores[target_position]:
            break
    th = target_scores[target_position]
    eer = target_position * 1.0 / target_size
    return eer, th


