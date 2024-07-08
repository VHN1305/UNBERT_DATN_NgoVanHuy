import pandas as pd
import numpy as np
from src.eval import calculate_mrr, calculate_ndcg, func, rank_func
import multiprocessing
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import os
from time import gmtime, strftime

def dcg_score(y_true, y_score, k=10):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2 ** y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_score, k=10):
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_score, k)
    return actual / best

def mrr_score(y_true, y_score):
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)

def concate_dateset():
    dev_score_1 = pd.read_csv("output/bge_mean/epoch2/dev_score_1.tsv", sep="\t")
    dev_score_2 = pd.read_csv("output/bge_mean/epoch2/dev_score_2.tsv", sep="\t")
    dev_score_3 = pd.read_csv("output/bge_mean/epoch2/dev_score_3.tsv", sep="\t")
    concate_dev_score = pd.concat([dev_score_1, dev_score_2, dev_score_3], ignore_index=True)
    print(concate_dev_score.info())
    concate_dev_score.to_csv("output/bge_mean/epoch2/dev_score.tsv", sep="\t", index=False)


if __name__ == "__main__":
    # concate_dateset()
    path = "output/both_level/epoch1/"
    EVAL_DF = pd.read_csv(path + "dev_score.tsv", sep="\t")
    labels = EVAL_DF["label"].values
    scores = EVAL_DF["score"].values
    impression_id = EVAL_DF["impression_id"].values

    groups_iter = EVAL_DF.groupby("impression_id")

    imp, df_groups = zip(*groups_iter)
    pool = multiprocessing.Pool()
    result = pool.map(func, df_groups)
    pool.close()
    pool.join()
    auc = np.mean(result)

    pool2 = multiprocessing.Pool()
    result2 = pool2.map(rank_func, df_groups)
    pool2.close()
    pool2.join()
    ranks = [r["rank"] for r in result2 if isinstance(r, dict) and "rank" in r]
    ranks_list = [item for sublist in ranks for item in sublist]
    EVAL_DF["rank"] = ranks_list
    EVAL_DF.to_csv("final_dev_score_rank.tsv", sep="\t", index=False)
    print("ranks_list", ranks_list)
    index = 0
    imps = [r["imp"] for r in result2]
    sum_auc = 0
    sum_mrr = 0
    sum_ndcg_5 = 0
    sum_ndcg_10 = 0
    for rank in tqdm(ranks, desc="Calculating metrics: "):
        label = labels[index:index+len(rank)]
        score = scores[index:index+len(rank)]
        sum_auc = roc_auc_score(label, score)
        sum_mrr += mrr_score(label, score)
        sum_ndcg_5 += ndcg_score(label, score, 5)
        sum_ndcg_10 += ndcg_score(label, score, 10)
        index += len(rank)

    mrr = sum_mrr / len(ranks)
    ndcg_5 = sum_ndcg_5 / len(ranks)
    ndcg_10 = sum_ndcg_10 / len(ranks)

    print(f"AUC: {format(auc, '.4f')}", end=", ")
    print(f"MRR: {format(mrr, '.4f')}", end=", ")
    print(f"NDCG@5: {format(ndcg_5, '.4f')}", end=", ")
    print(f"NDCG@10: {format(ndcg_10, '.4f')}")
    print(f"{format(auc, '.4f')} & {format(mrr, '.4f')} & {format(ndcg_5, '.4f')} & {format(ndcg_10, '.4f')}")
    log_file = os.path.join(path, "{}-{}-{}.log".format(
        "dev", "small", strftime('%Y%m%d%H%M%S', gmtime())))
    os.makedirs(path, exist_ok=True)


    def printzzz(log):
        with open(log_file, "a") as fout:
            fout.write(log + "\n")
        print(log)

    printzzz("dev AUC: {:.4f}".format(auc)
             + " MRR: {:.4f}".format(mrr)
             + " NDCG@5: {:.4f}".format(ndcg_5)
             + " NDCG@10: {:.4f}".format(ndcg_10)
             )
    printzzz("dev success!")

