# encoding=utf-8
# 首先获取每个指控或者法律条款对应的case descs
# 计算每个指控或者法律条款对应的tf-idf向量
import json
import thulac
import utils
import jieba
import os
import pickle
import math
import torch

def idx2cases(path):
    accu2cases = {}
    art2cases = {}
    penalty2cases = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            acc = item[1]
            art = item[2]
            penalty = item[3]
            if acc not in accu2cases:
                accu2cases[acc] = [item[0]]
            else:
                accu2cases[acc].append(item[0])
            if art not in art2cases:
                art2cases[art] = [item[0]]
            else:
                art2cases[art].append(item[0])
            if penalty not in penalty2cases:
                penalty2cases[penalty] = [item[0]]
            else:
                penalty2cases[penalty].append(item[0])

    return accu2cases, art2cases, penalty2cases

def dist_in_cate(docs, token):
    # token 在类别c的文档中的分布
    doc_count = 0
    for i in range(len(docs)):
        if token in docs[i]:
            doc_count+=1
    return doc_count/(len(docs)+0.0001)

def freq_in_cate(docs, token):
    freq = 0.
    total_token = 0.
    for d in docs:
        freq += d.count(token)
        total_token += len(d)
    return freq/(total_token+0.001)

def idf(dic,token):
    cate_num = len(dict.keys())
    total_dist = 0.
    for key,values in dict.items():
        total_dist += dist_in_cate(values, token)
    return math.log(cate_num/total_dist,10)

def tf_idf(dic, lang):
    # tf-idf向量
    idx2tf_idf = {}
    for key, values in dic.items():
        # 初始化向量表示
        tf_idf_rep = [0]*(lang.n_words-1)
        for i in range(lang.n_words-1):
            token = lang.index2word[i]
            # 计算tf_idf值
            item_1 = dist_in_cate(values, token)
            item_2 = freq_in_cate(values, token)
            item_3 = idf(dic,token)
            tf_idf_rep[i] = item_1*item_2*item_3
        idx2tf_idf[key] = tf_idf_rep

data_paths = ["dataset/CAIL-SMALL"]



print("加载语料库信息...")
with open() as f:
    pickle.loads(f)
paths = ["dataset/CAIL-SMALL"]
for path in paths:
    acc2cases, art2cases, penalty2cases = idx2cases(path)


