def compute_same_diff_from_label(label1, label2):
    subjnum1 = label1[:, 0]
    scannum1 = label1[:, 1]
    subjnum2 = label2[:, 0]
    scannum2 = label2[:, 1]
    same = (subjnum1[:, None] == subjnum2[None, :]) & (scannum1[:, None] != scannum2[None, :])
    diff = (subjnum1[:, None] != subjnum2[None, :])
    return same, diff