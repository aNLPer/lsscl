# encoding=utf-8
# 首先获取每个指控或者法律条款对应的case descs
# 计算每个指控或者法律条款对应的tf-idf向量
import json
import pickle
import math
import time 
import numpy as np

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

    return (accu2cases, art2cases, penalty2cases)

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
    return freq/(total_token+0.0001)

def idf(dic,token):
    cate_num = len(dic.keys())
    total_dist = 0.0001
    for key,values in dic.items():
        total_dist += dist_in_cate(values, token)
    return math.log(cate_num/total_dist,10)

def tf_idf(dic, lang):
    # tf-idf向量
    idx2tf_idf = {}
    for key, values in dic.items():
        # 初始化向量表示
        tf_idf_rep = [0]*(lang.n_words-1)
        for i in range(1,lang.n_words):
            token = lang.index2word[i]
            # 计算tf_idf值
            item_1 = dist_in_cate(values, token)
            item_2 = freq_in_cate(values, token)
            item_3 = idf(dic,token)
            tf_idf_rep[i] = item_1*item_2*item_3
        idx2tf_idf[key] = tf_idf_rep
    return idx2tf_idf

def contrust_graph(dic,threshold):
    # 相似矩阵
    keys = list(dic.keys())
    sim_matrix = np.zeros(shape=(len(keys), len(keys)))
    arr = np.array([v for v in dic.values()])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            cos_sim = arr[i].dot(arr[j])/(np.linalg.norm(arr[i])*np.linalg.norm(arr[j]))
            if cos_sim>threshold:
                sim_matrix[i][j] = cos_sim
            else:
                sim_matrix[i][j] = 0.
    
    # 根据相似矩阵得到label_sim_graph
    sim_graph = {}
    for k in keys:
        sim_graph.setdefault(k,{})
    for i in range(len(keys)):
        for j in range(len(keys)):
            if i == j:
                continue
            if sim_matrix[i][j]!=0:
                sim_graph[keys[i]][keys[j]]=sim_matrix[i][j]
    return sim_graph

def test_contrust_graph():
    d = {"a":[1,2,3,4], "b":[1,2,3,4], "c":[-1,-2,-3,4], "d":[-1,-2,-3,4], "e":[1,2,3,4]}
    print(list(d.keys()))
    m, g = graph_construction.contrust_graph(d, 0.5)
    print(m, g)

if __name__=="__main__":
    data_paths = ["dataset/CAIL-SMALL/train_seg.txt","dataset/CAIL-LARGE/train_seg.txt"]
    lang_paths = ["dataset/CAIL-SMALL/train_seg_lang.pkl", "dataset/CAIL-LARGE/train_seg_lang.pkl"]
    threshold = 0.5
    label_reps = []
    for i in range(len(data_paths)):
        print("加载语料库信息...")
        with open(lang_paths[i], "rb") as f:
            lang = pickle.load(f)
        print("获取label-case字典...")
        dicts = idx2cases(data_paths[i])
        print("为每个lebel计算tf-idf表示向量...")
        start = time.time()
        for d in dicts:
            rep = tf_idf(d, lang)
            label_reps.append(rep)
        print(f"{round((time.time()-start)/60, 6)}min")
        print("为label构造相似图...")
        
        
        





