import numpy as np
import json
import pandas as pd
from sklearn.metrics import classification_report
from tqdm import tqdm
import os


def gen_gt_dict(df):
    d = {}
    for idx in range(len(df)):
        d[df.loc[idx, "qid"]] = {
            "qidx": df.loc[idx, "qidx"],
            "did": eval(df.loc[idx, "did"]),
            "didx": eval(df.loc[idx, "didx"]),
            "label": eval(df.loc[idx, "label"])
        }

    return d


def eval_for_outputs(typ, dataset, outputs_file, topk=50):
    gt_type = typ.replace('_updated', '').replace('_eval', '')
    dataset_split_dir = '../dataset/{}/splits/data'.format(dataset)
    gt_file = os.path.join(dataset_split_dir, 'top{}.{}'.format(topk, gt_type))
    gt = gen_gt_dict(pd.read_csv(gt_file, sep="\t", names=[
                     "qid", "qidx", "did", "didx", "label"]))

    wrong_cls_cases = []
    ranking_res = []
    classifying_ans = np.array([], dtype=int)
    classifying_pred = np.array([], dtype=int)
    ranking_ans = []
    ranking_pred = []

    out = json.load(open(outputs_file, 'r'))

    for o in out:
        # score: [did, 0_class_score, 1_class_score]
        qid, score = o[0], o[1]
        pred = [0 if (x[1] >= x[2]) else 1 for x in score]
        ans = gt[qid]["label"]
        assert len(pred) == len(ans)

        wrong_cls_cases.append(
            [(qid, score[i][0], ans[i], pred[i]) for i in range(len(pred)) if ans[i] != pred[i]])
        rans = [gt[qid]['did'][i] for i, l in enumerate(ans) if l == 1]
        rpred = [d for d, _, _ in sorted(
            score, key=lambda x: x[2], reverse=True)]

        classifying_ans = np.append(classifying_ans, np.array(ans))
        classifying_pred = np.append(classifying_pred, pred)

        ranking_ans.append(rans)
        ranking_pred.append(rpred)
        ranking_res.append((qid, rans, rpred))

    class_report = compute_classification_metrics(
        classifying_ans, classifying_pred)
    ranking_report = compute_ranking_metrics(ranking_ans, ranking_pred)

    res_file = outputs_file.replace('_outputs_', '_res_')
    wrong_cls_file = outputs_file.replace('_outputs_', '_wrong_cls_')
    ranking_file = outputs_file.replace('_outputs_', '_rankings_')

    json.dump({
        'classification': class_report,
        'ranking': ranking_report
    }, open(res_file, 'w'), indent=4)
    json.dump(wrong_cls_cases, open(wrong_cls_file, 'w'), indent=4)
    json.dump(ranking_res, open(ranking_file, 'w'), indent=4)


def compute_classification_metrics(ans, pred):
    return classification_report(ans, pred, target_names=[0, 1], digits=4, output_dict=True)


def compute_ranking_metrics(ans, pred, digits=True):
    '''
    Input:  ans: [[ans1_articleID1, ans1_articleID2, ...],
                 [ans2_articleID1, ans2_articleID2, ...],
                 ...
                ]
            pred:[[pred1_articleID1, pred1_aticleID2, ...],
                 [pred2_articleID1, pred2_aticleID2, ...],
                 ...
                ] (Ordered)
            metric: {'MRR', 'MAP', 'HasPositives'}
            k: Metric@ Top-k. Only for MAP and HasPositives. default = [1, 3, 5, 10, 20, 50 ,500, 1000]
    Example: ans1_articleID1: '5f5e37e063fc4d00ff6be951'

    Output: dict of scores of the samples
    '''

    if len(ans) == 0 or len(pred) == 0:
        return 0

    ranks = []
    for i in tqdm(range(len(pred))):
        rank = []
        for j in range(len(pred[i])):
            if pred[i][j] in ans[i]:
                rank.append(j + 1)
        if len(ans[i]) - len(rank) > 0:
            rank.extend([0] * (len(ans[i]) - len(rank)))
        ranks.append(rank)

    if ranks == []:
        return 0

    # print(ranks)

    result = {
        'MRR': MRR(ranks)
    }
    for metric in ['MAP', 'HasPositives', 'MAP_updated']:
        for k in [1, 3, 5, 10, 20, 50]:
            # for k in [1, 3, 5]:
            result[metric + '@' + str(k)] = eval(metric +
                                                 '(ranks,' + str(k) + ')')

    if not digits:
        for k in result.keys():
            result[k] = '{:.3f}'.format(result[k])

    return result


def MRR(ranks):
    rr = []
    for rank in ranks:
        r = 0
        for rk in rank:
            if rk > 0:
                r = rk
                break
        rr.append(1 / r if r != 0 else 0)

    mrr = np.mean(rr)
    return mrr


def MAP(ranks, k):
    APs = []
    for rank in ranks:
        AP = 0
        i = 1

        for r in rank:
            if r > 0 and r <= k:
                AP += i / r
                i += 1
        if len(rank) == 0:
            AP = 0
        else:
            AP /= min(len(rank), k)
        APs.append(AP)

    return np.mean(APs)


def MAP_updated(ranks, k):
    APs = []
    for rank in ranks:
        AP = 0
        i = 1

        for r in rank:
            if r > 0 and r <= k:
                AP += i / r
                i += 1
        if len(rank) == 0:
            AP = 0
        else:
            # AP /= min(len(rank), k)
            AP /= len(rank)
        APs.append(AP)

    return np.mean(APs)


def HasPositives(ranks, k):
    Has = []
    for rank in ranks:
        if len(rank) > 0 and rank[0] > 0:
            if rank[0] <= k:
                Has.append(1)
                continue
        Has.append(0)

    return np.mean(Has)
