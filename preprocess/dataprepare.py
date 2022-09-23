# coding:utf-8
import sys,os

# sys.path.append('需要作为模块引入的路径')
# 添加当前路径的前一级文件作为源文件夹
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import configparser
import os
import numpy as np
import pickle
import re
import json
import thulac
import utils
from utils import Lang
from transformers import BertModel, BertTokenizer
import numpy as np
import operator

class DataPreprocess():
    def __init__(self, dataset_base_path, folders, file_names):
        self.dataset_base_path = dataset_base_path # 数据集根目录
        self.folders = folders  # 文件夹名称
        self.file_names = file_names  # 文件名称

    # filter cases 
    def case_filter(self):
        """
        根据文本长度、样本频率过滤数据集
        :param source_path:
        :param target_path:
        :return:
        """
        for folder in self.folders:
            dict_articles = {}  # 法律条款：数量
            dict_accusations = {}  # 指控：数量
            for fn in self.file_names:
                print(f'{folder}/{fn}.json statistic beginning')
                with open(os.path.join(self.dataset_base_path, folder, f"{fn}.json"), 'r', encoding='utf-8') as f:
                    for line in f:
                        example = json.loads(line)
                        example_articles = example['meta']['relevant_articles']
                        example_accusation = example['meta']['accusation']
                        example_fact = example['fact']
                        # 仅统计单条款、单指控、仅一审的案件的指控和条款
                        if len(example_articles) == 1 and \
                                len(example_accusation) == 1 and \
                                '二审' not in example_fact and \
                                len(example_fact) > 10:
                            if dict_articles.__contains__(example_articles[0]):
                                dict_articles[example_articles[0]] += 1
                            else:
                                dict_articles.update({example_articles[0]: 1})
                            if dict_accusations.__contains__(example_accusation[0]):
                                dict_accusations[example_accusation[0]] += 1
                            else:
                                dict_accusations.update({example_accusation[0]: 1})
                print(f'{folder}/{fn}.json statistic over')

            # 过滤掉频次小于100的条款和指控
            dict_articles = utils.filter_dict(dict_articles)
            dict_accusations = utils.filter_dict(dict_accusations)

            articles_sum = utils.sum_dict(dict_articles)
            accusation_sum = utils.sum_dict(dict_accusations)

            print('filter begining......')
            while articles_sum != accusation_sum:
                dict_accusations = utils.reset_dict(dict_accusations)
                dict_articles = utils.reset_dict(dict_articles)
                for fn in self.file_names:
                    with open(os.path.join(self.dataset_base_path, folder, f"{fn}.json"), 'r', encoding='utf-8') as f:
                        for line in f:
                            example = json.loads(line)
                            example_articles = example['meta']['relevant_articles']
                            example_accusation = example['meta']['accusation']
                            example_fact = example['fact']
                            if len(example_articles) == 1 and \
                                    len(example_accusation) == 1 and \
                                    '二审' not in example_fact and \
                                        len(example_fact) > 10:
                                # 该案件对应的article和accusation频率都大于100
                                if dict_articles.__contains__(example_articles[0]) and \
                                        dict_accusations.__contains__(example_accusation[0]): 
                                    dict_articles[example_articles[0]] += 1
                                    dict_accusations[example_accusation[0]] += 1
                                else:
                                    continue
                dict_articles = utils.filter_dict(dict_articles)
                dict_accusations = utils.filter_dict(dict_accusations)

                articles_sum = utils.sum_dict(dict_articles)
                accusation_sum = utils.sum_dict(dict_accusations)

                print('articles_num: ' + str(len(dict_articles)))
                print('article_sum: ' + str(articles_sum))

                print('accusation_num=' + str(len(dict_accusations)))
                print('accusation_sum: ' + str(accusation_sum))
                print("\n\n")

            for fn in self.file_names:
                with open(os.path.join(self.dataset_base_path, folder, f"{fn}_filtered.json"), "w", encoding="utf-8") as fw:
                    with open(os.path.join(self.dataset_base_path, folder, f"{fn}.json"), 'r', encoding='utf-8') as f:
                        for line in f:
                            example = json.loads(line)
                            example_articles = example['meta']['relevant_articles']
                            example_accusation = example['meta']['accusation']
                            example_fact = example['fact']
                            if len(example_articles) == 1 and \
                                    len(example_accusation) == 1 and \
                                    '二审' not in example_fact and \
                                    len(example_fact) > 10:
                                # 该案件对应的article和accusation频率都大于100
                                if dict_articles.__contains__(example_articles[0]) and \
                                        dict_accusations.__contains__(example_accusation[0]):
                                    fw.write(line)
                                else:
                                    continue
            print('filter over......')

    # process special charac
    def charc_process(self):
        '''
        构造数据集：[case_desc, "acc", "article","penalty"]
        # 分词
        # 去除特殊符号
        # 去除停用词
        # 去除标点
        # 去除停用词和标点
        # 同类别的accusation(样本+4)
        :param case_path: 案件描述文件
        :param acc2desc: 指控：指控描述 （字典）
        :return: [[[case_desc,case_desc,...], "acc", "acc_desc"],]
        '''
        # 加载分词器
        thu = thulac.thulac(user_dict="preprocess/Thuocl_seg.txt", seg_only=True)
        # 加载停用词表
        stopwords = []
        for n in os.listdir("preprocess/stopwords"):
            stopwords.extend(utils.get_filter_symbols(os.path.join("preprocess/stopwords", n)))
        stopwords = list(set(stopwords))
        # 加载标点
        punctuations = utils.get_filter_symbols("preprocess/punctuation.txt")
        # 加载特殊符号
        special_symbols = utils.get_filter_symbols("preprocess/special_symbol.txt")

        for folder in self.folders:
            for fn in self.file_names:
                print(f"start processing data {folder}/{fn}_filtered.json")
                with open(os.path.join(self.dataset_base_path, folder, f"{fn}_seg.txt"), "w", encoding="utf-8") as fw:
                    count = 0
                    with open(os.path.join(self.dataset_base_path, folder, f"{fn}_filtered.json"), "r", encoding="utf-8") as f:
                        for line in f:
                            count += 1
                            item = [] # 单条训练数据
                            example = json.loads(line)
                            example_fact = example["fact"]
                            
                            pattern = re.compile(r"[\r\n]")
                            example_fact = pattern.sub("", example_fact)
                            # 过滤law article内容
                            # example_fact = utils.filterStr(example["fact"])

                            # if folder == "CAIL-LARGE":
                            #     example_fact = example_fact.strip()
                            #     pattern = re.compile(r"\n")
                            #     content = pattern.search(example_fact)
                            #     if content is not None:
                            #         content_span = content.span()
                            #         example_fact = example_fact[:content_span[0]].strip()

                            # 去除特殊符号
                            example_fact = [char for char in example_fact if char not in special_symbols]
                            example_fact = "".join(example_fact)

                            # # 删除过短文本
                            # if len(example_fact) < 15:
                            #     continue

                            # 分词
                            example_fact_seg = [word.strip() for word in thu.cut(example_fact, text=True).split(" ")]
                            # 处理时间
                            example_fact_seg = [re.sub(r"\d+?[年月日时点分]", "num", word) for word in example_fact_seg]
                            example_fact_seg = [word for word in example_fact_seg
                                                if word not in ["num", "下午", "上午", "早上", "凌晨", "晚", "晚上", "许"] and "num" not in word]

                            # 去除标点
                            # example_fact_seg = [word for word in example_fact_seg if word not in punctuations]
                            # example_fact_seg = "".join(example_fact_seg)
                            
                            # 删除过短文本
                            if len(example_fact_seg) < 10:
                                continue

                            item.append(example_fact_seg)

                            # 去除停用词
                            # example_fact_seg = [word for word in example_fact_seg if word not in stopwords]

                            # 添加标签
                            example_accu = example["meta"]['accusation'][0]
                            example_accu = example_accu.replace("[", "")
                            example_accu = example_accu.replace("]", "")
                            example_art = int(example["meta"]['relevant_articles'][0])
                            item.append(example_accu)
                            item.append(example_art)
                            example_penalty = example["meta"]["term_of_imprisonment"]
                            if (example_penalty["death_penalty"] == True or example_penalty["life_imprisonment"] == True):
                                item.append(0)
                            elif example_penalty["imprisonment"] > 10 * 12:
                                item.append(1)
                            elif example_penalty["imprisonment"] > 7 * 12:
                                item.append(2)
                            elif example_penalty["imprisonment"] > 5 * 12:
                                item.append(3)
                            elif example_penalty["imprisonment"] > 3 * 12:
                                item.append(4)
                            elif example_penalty["imprisonment"] > 2 * 12:
                                item.append(5)
                            elif example_penalty["imprisonment"] > 1 * 12:
                                item.append(6)
                            elif example_penalty["imprisonment"] > 9:
                                item.append(7)
                            elif example_penalty["imprisonment"] > 6:
                                item.append(8)
                            elif example_penalty["imprisonment"] > 0:
                                item.append(9)
                            else:
                                item.append(10)
                            # 指控描述
                            list_str = json.dumps(item, ensure_ascii=False)
                            fw.write(list_str+"\n")
                            if count%5000==0:
                                print(f"已有{count}条数据被处理")
    
    # 过滤掉训练数据少于阈值的case
    def data_filter(self, acc, art):
        for folder in self.folders:
            for fname in self.file_names:
                print(folder,f"/{fname}")
                fw = open(os.path.join(self.dataset_base_path, folder, f"{fname}.txt"), "w", encoding="utf-8")
                with open(os.path.join(self.dataset_base_path, folder, f"{fname}_seg.txt"), "r", encoding="utf-8") as f:
                    for line in f:
                        item = json.loads(line)
                        if item[1] not in acc and item[2] not in art:
                            fw.write(line)
        fw.close()
        print("end...")
    
    # 数据统计并过滤掉训练数据少于阈值的case
    def dataset_statistic(self):
        with open("label_sim_graph_construction/charge_sim_graph.pkl", "rb") as f:
            graph = pickle.load(f)

        train_accu2casenum = {}
        train_article2casenum = {}
        test_accu2casenum = {}
        test_article2casenum = {}
        with open(os.path.join(self.dataset_base_path, "CAIL-SMALL", "train_seg.txt"), "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item[1] not in train_accu2casenum:
                    train_accu2casenum[item[1]] = 1
                else:
                    train_accu2casenum[item[1]] += 1
                if item[2] not in train_article2casenum:
                    train_article2casenum[item[2]] = 1
                else:
                    train_article2casenum[item[2]] += 1
        with open(os.path.join(self.dataset_base_path, "CAIL-SMALL", "test_seg.txt"), "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item[1] not in test_accu2casenum:
                    test_accu2casenum[item[1]] = 1
                else:
                    test_accu2casenum[item[1]] += 1
                if item[2] not in test_article2casenum:
                    test_article2casenum[item[2]] = 1
                else:
                    test_article2casenum[item[2]] += 1


        # # 测试集和训练集标签差集
        # test_acc = set(list(test_accu2casenum.keys()))
        # train_acc = set(list(train_accu2casenum.keys()))
        # test_art = set(list(test_article2casenum.keys()))
        # train_art = set(list(train_article2casenum.keys()))
        # print(train_acc-test_acc)
        # print(train_art-test_art)

        # 排序
        tr_accu2casenum = sorted(list(train_accu2casenum.items()), key=lambda x: x[1])            
        te_accu2casenum = sorted(list(test_accu2casenum.items()), key=lambda x: x[1])            
        tr_article2casenum = sorted(list(train_article2casenum.items()), key=lambda x: x[1])            
        te_article2casenum = sorted(list(test_article2casenum.items()), key=lambda x: x[1])            
        
        return train_accu2casenum, train_article2casenum, test_accu2casenum, test_article2casenum
    
    # statistic corpus
    def getLang(self):
        file_name = ["train_seg"]
        for folder in self.folders:
            for fn in file_name:
                lang = Lang()
                with open(f"dataset/{folder}/{fn}_lang.pkl", "wb") as lang_f:
                    print(f"processing {folder}/{fn}")
                    with open(os.path.join(self.dataset_base_path, folder, f"{fn}.txt"), "r", encoding="utf-8") as f:
                        for line in f:
                            sample = json.loads(line)
                            lang.addSentence(sample[0])
                            lang.addLabel(sample[1], sample[2])
                    lang.update_label2index()
                    pickle.dump(lang, lang_f)
        print("end...")

    def statistic_sen_length(self):
        length_500 = 0
        with open("dataset/CAIL-LARGE/train_seg.txt", "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if len(item[0])<500:
                    length_500+=1
        print(length_500)
if __name__=="__main__":
    dp = DataPreprocess(dataset_base_path="dataset", folders=["CAIL-SMALL","CAIL-LARGE"], file_names=["test", "train"])
    dp.dataset_statistic()
    # dp.charc_process()
    # dp.segment()
    # dp.data_filter(acc=[], art=[356])
    # dp.case_filter()
    # dp.getLang()
    # dp.statistic_sen_length()
    acc = ['容留他人吸毒', '动植物检疫徇私舞弊','单位受贿', '对单位行贿','妨害作证']
    
    










