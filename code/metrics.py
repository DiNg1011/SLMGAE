import codecs
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, roc_curve, precision_recall_curve


def evalution(adj_rec, train_pos, test_pos):
    num = adj_rec.shape[0]
    x, y = np.triu_indices(num, k=1)

    c_set = set(zip(x, y)) - \
            set(zip(train_pos[:, 0], train_pos[:, 1])) - set(zip(train_pos[:, 1], train_pos[:, 0]))

    inx = np.array(list(c_set))
    Y = np.zeros((num, num))
    Y[test_pos[:, 0], test_pos[:, 1]] = 1
    Y[test_pos[:, 1], test_pos[:, 0]] = 1
    labels = Y[inx[:, 0], inx[:, 1]]
    val = adj_rec[inx[:, 0], inx[:, 1]]

    fpr, tpr, throc = roc_curve(labels, val)

    auc_val = auc(fpr, tpr)
    prec, rec, thpr = precision_recall_curve(labels, val)
    aupr_val = auc(rec, prec)

    # np.save('pred.npy', val)
    # np.save('fpr.npy', fpr)
    # np.save('tpr.npy', tpr)
    # np.save('prec.npy', prec)
    # np.save('rec.npy', rec)

    f1_val = 0
    for i in range(len(prec)):
        if (prec[i] + rec[i]) == 0:
            continue
        f = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
        if f > f1_val:
            f1_val = f

    return auc_val, aupr_val, f1_val


def evalution_bal(adj_rec, edges_pos, edges_neg):
    # Predict on test set of edges
    preds = []
    for e in edges_pos:
        preds.append(adj_rec[e[0], e[1]])

    preds_neg = []

    for e in edges_neg:
        preds_neg.append(adj_rec[e[0], e[1]])
        if len(preds_neg) == len(preds):
            break

    preds_all = np.hstack((preds, preds_neg))
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])

    fpr, tpr, _ = roc_curve(labels_all, preds_all)
    roc_score = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(labels_all, preds_all)
    aupr_score = auc(rec, prec)

    f1_val = 0
    for i in range(len(prec)):
        if (prec[i] + rec[i]) == 0:
            continue
        f = 2 * prec[i] * rec[i] / (prec[i] + rec[i])
        if f > f1_val:
            f1_val = f

    # for x, y in zip(fpr, tpr):
    #     roc = str(x) + '\t' + str(y)
    #     codecs.open('roc_bal.txt', mode='a', encoding='utf-8').write(roc + "\n")
    #
    # for x, y in zip(prec, rec):
    #     prc = str(x) + '\t' + str(y)
    #     codecs.open('prc_bal.txt', mode='a', encoding='utf-8').write(prc + "\n")

    return roc_score, aupr_score, f1_val
