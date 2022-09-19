# encoding=utf-8
# 首先获取每个指控或者法律条款对应的case descs
# 计算每个指控或者法律条款对应的tf-idf向量
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import json
import pickle
import math
import time 
import numpy as np
select_size = 50
threshold = 0.5

def data_pre(path):
    """
    path:数据集路径
    """
    accu2cases = {}
    art2cases = {}
    penalty2cases = {}
    accu_lang = utils.Lang()
    art_lang = utils.Lang()
    penalty_lang = utils.Lang()
    print("data loading ...")
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
    # 选取数据
    print("data preparing ...")
    for d in [accu2cases, art2cases, penalty2cases]:
        for k,vs in d.items():
            if len(vs) < select_size:
                continue
            idxs = np.random.choice(range(len(vs)), size=select_size, replace=False)
            d[k] = [vs[i] for i in idxs]
    print("collecting corpu's infor ...")
    for d,lang in zip([accu2cases, art2cases, penalty2cases],[accu_lang, art_lang, penalty_lang]):
        for k, vs in d.items():
            for s in vs:
                lang.addSentence(s)
    # 选取的数据集和其对应的
    return zip((accu2cases, art2cases, penalty2cases),(accu_lang, art_lang, penalty_lang))

def dist_in_cate(docs, token):
    # 返回包含token的文档数目/文档总数
    doc_count = 0
    for i in range(len(docs)):
        if token in docs[i]:
            doc_count+=1
    return doc_count/(len(docs)+0.0001)

def freq_in_cate(docs, token):
    # token在某类文档中出现的次数/某类别包含的总词汇数
    freq = 0.
    total_token = 0.
    for d in docs:
        freq += d.count(token)
        total_token += len(d)
    return freq/(total_token+0.0001)

def idf(dic,token):
    # 类别总数/包含该token的类别数目
    cate_num = len(dic.keys())
    total_dist = 0.0001
    for key,values in dic.items():
        total_dist += dist_in_cate(values, token)
    return math.log(cate_num/total_dist,2)

def tf_idf(dic, lang):
    # tf-idf向量
    idx2tf_idf = {}
    for key, values in dic.items():
        # 初始化向量表示
        start = time.time()
        tf_idf_rep = [0]*(lang.n_words-1)
        for i in range(1,lang.n_words):
            # start = time.time()
            token = lang.index2word[i]
            # 计算tf_idf值
            item_1 = dist_in_cate(values, token)
            item_2 = freq_in_cate(values, token)
            item_3 = idf(dic,token)
            tf_idf_rep[i-1] = round(item_1*item_2*item_3, 6)
            # print(f"time: {time.time()-start}")
        idx2tf_idf[key] = tf_idf_rep
        print(f"time: {round((time.time()-start)/60, 2)}")
        print(f"{key} : {tf_idf_rep[:10]}")
    return idx2tf_idf

def contrust_graph(dic,threshold):
    # 相似矩阵
    keys = list(dic.keys())
    sim_matrix = np.zeros(shape=(len(keys), len(keys)))
    arr = np.array([v for v in dic.values()])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[0]):
            cos_sim = arr[i][1:].dot(arr[j][1:])/(np.linalg.norm(arr[i][1:])*np.linalg.norm(arr[j][1:]))
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
    # data_zip = data_pre(path="dataset/CAIL-SMALL/train_seg.txt")
    # label_reps = []
    # print("calculating tf_idf reps...")
    # for d, l in data_zip:
    #     key2tf_idf = tf_idf(d,l)
    #     label_reps.append(key2tf_idf)
    # with open(f"{select_size}_small_label_reps.pkl", "wb") as f:
    #      pickle.dump(label_reps, f)
    
    with open("label_sim_graph_construction/32_small_label_reps.pkl", "rb") as f:
        reps = pickle.load(f)
    for name, rep in zip(["charge_sim_graph", "article_sim_graph", "penalty_sim_graph"], reps):
        graph = contrust_graph(rep, threshold=threshold)
        with open(f"label_sim_graph_construction/{name}.pkl","wb") as f:
            pickle.dump(graph, f)
    with open("label_sim_graph_construction/charge_sim_graph.pkl", "rb") as f:
        graph = pickle.load(f)
    print(graph)
    # data_paths = ["dataset/CAIL-SMALL/train_seg.txt","dataset/CAIL-LARGE/train_seg.txt"]
    # lang_paths = ["dataset/CAIL-SMALL/train_seg_lang.pkl", "dataset/CAIL-LARGE/train_seg_lang.pkl"]
    # threshold = 0.5
    # label_reps = []
    # for i in range(len(data_paths)):
    #     print("加载语料库信息...")
    #     with open(lang_paths[i], "rb") as f:
    #         lang = pickle.load(f)
    #     print("获取label-case字典...")
    #     dicts = idx2cases(data_paths[i])
    #     print("为每个lebel计算tf-idf表示向量...")
    #     start = time.time()
    #     for d in dicts:
    #         rep = tf_idf(d, lang)
    #         label_reps.append(rep)
    #     print(f"{round((time.time()-start)/60, 6)} min")
    #     with open("label_reps.pkl", "wb") as f:
    #         pickle.dump(label_reps, f)
    #     print("为label构造相似图...")
        
        
        





