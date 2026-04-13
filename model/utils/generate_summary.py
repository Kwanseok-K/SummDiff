# -*- coding: utf-8 -*-
import numpy as np
#from knapsack import knapsack_ortools
from model.utils.knapsack_implementation import knapSack
import math
import csv
from collections import Counter
# from knapsack_implementation import knapSack

def generate_summary(ypred, cps, n_frames, nfps, positions, proportion=0.15, method='knapsack'):
    """Generate keyshot-based video summary i.e. a binary vector.
    Args:
    ---------------------------------------------
    - ypred: predicted importance scores.
    - cps: change points, 2D matrix, each row contains a segment.
    - n_frames: original number of frames.
    - nfps: number of frames per segment.
    - positions: positions of subsampled frames in the original video.
    - proportion: length of video summary (compared to original video length).
    - method: defines how shots are selected, ['knapsack', 'rank'].
    """

    n_segs = len(cps)
    n_frames = n_frames[0]
    frame_scores = np.zeros((n_frames), dtype=np.float32)
    # if positions.dtype != int:
    #     positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i+1]
        if i == len(ypred):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = ypred[i]
    # print("frame_scores")
    # print(frame_scores)
    # print(frame_scores.shape)
    seg_score = []
    for seg_idx in range(n_segs):
        start, end = int(cps[seg_idx][0]), int(cps[seg_idx][1]+1)
        # print("start ", start, " end ", end)
        scores = frame_scores[start:end]
        # print("scores")
        # print(scores)
        # print(len(scores))
        seg_score.append(float(scores.mean()))

    limits = int(math.floor(n_frames * proportion))
    # print("limits")
    # print(limits)
    # print("seg_score")
    # print(seg_score)
    # print(len(seg_score))
    picks = knapSack(limits, nfps, seg_score, len(nfps))

    summary = np.zeros((1), dtype=np.float32) # this element should be deleted
    for seg_idx in range(n_segs):
        nf = nfps[seg_idx]
        if seg_idx in picks:
            tmp = np.ones((nf), dtype=np.float32)
        else:
            tmp = np.zeros((nf), dtype=np.float32)
        summary = np.concatenate((summary, tmp))

    summary = np.delete(summary, 0) # delete the first element
    return summary

def get_gt(downsampled=True):
    annot_path = f"./dataset/ydata-anno.tsv"
    with open(annot_path) as annot_file:
        annot = list(csv.reader(annot_file, delimiter="\t"))
    annotation_length = list(Counter(np.array(annot)[:, 0]).values())
    user_scores = []
    for idx in range(1,51):
        init = (idx - 1) * annotation_length[idx-1]
        till = idx * annotation_length[idx-1]
        user_score = []
        for row in annot[init:till]:
            curr_user_score = row[2].split(",")
            curr_user_score = np.array([float(num) for num in curr_user_score])
            curr_user_score = curr_user_score / curr_user_score.max(initial=-1)
            if downsampled:
                curr_user_score = curr_user_score[::15]

            user_score.append(curr_user_score)
        user_scores.append(user_score)
    return user_scores