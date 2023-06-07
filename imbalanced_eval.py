import json
import numpy as np
import re

def preprogress(f):
    f = open(f)
    max_lenn = 76332
    line1 = f.readline()
    gt = []
    i = 0
    while i<max_lenn:
        line1 = line1.split( )
        try:
            if line1[0] == 'Impression:':
                line1 = line1[1:]
                gt.append(line1)
            i += 1
            line1 = f.readline()
        except:
            i += 1
            line1 = f.readline()
            pass
    f.close()
    return gt

gt = preprogress('/home/ywu10/Documents/multimodel/results/1run_gt_results_2022-05-04-17-20.txt')
pre = preprogress('/home/ywu10/Documents/multimodel/results/1run_pre_results_2022-05-06-07-28.txt')

def imbalanced_eval(pre,tgt,words):

    recall_ = []
    precision_ = []
    right_ = []
    gap = len(words)//7

    mm = 0
    for index in range(0,len(words),gap):
        mm += 1
        right = 0
        recall = 0
        precision = 0
        for i in range(len(tgt)):
            a = [j for j in tgt[i] if j in words[index:index+gap]]
            b = [j for j in pre[i] if j in words[index:index+gap]]
            right += min(len([j for j in a if j in b]),len([j for j in b if j in a]))
            recall += len(a)
            precision += len(b)
        recall_.append(recall)
        precision_.append(precision)
        right_.append(right)
    print(f'recall:{np.array(right_)/np.array(recall_)}')
    print(f'precision:{np.array(right_)/np.array(precision_)}')
    print(precision_)
    print(recall_)

imbalanced_eval(pre,gt,words)

