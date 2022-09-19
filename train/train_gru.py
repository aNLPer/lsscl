#coding:utf-8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import utils
import torch
import time
import pickle
import gensim
import configparser
from models import GRUBase
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from transformers import get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup, get_cosine_schedule_with_warmup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
corpus_info_path = ["dataset/CAIL-SMALL/lang.pkl", "dataset/CAIL-LARGE/lang.pkl"]
dataset_path = ["dataset/CAIL-SMALL", "dataset/CAIL-LARGE"]
pretrain_lm = "dataset/pretrain/law_token_vec_300.bin"

print("load model params...")
param = utils.Params("gru-base")

print("load pretrained word2vec...")
pretrained_w2c = gensim.models.KeyedVectors.load_word2vec_format(pretrain_lm, binary=False)

def train():
    for i in range(1):
        with open(corpus_info_path[i], "rb") as f:
            lang = pickle.load(f)
        print(f"load {lang.name}_corpus info...")

        print(f"load {dataset_path[i]} train data...")
        train_seq, train_charge_labels, train_article_labels, train_penalty_labels = \
                                                            utils.prepare_data(os.path.join(dataset_path[i], "train.txt"), 
                                                                                lang,
                                                                                max_length=param.MAX_LENGTH, 
                                                                                pretrained_vec=pretrained_w2c)
                                                                        
        print(f"load {dataset_path[i]} test data...")
        test_seq, test_charge_labels, test_article_labels, test_penalty_labels = \
                                                            utils.prepare_data(os.path.join(dataset_path[i], "test.txt"), 
                                                                                lang, 
                                                                                max_length=param.MAX_LENGTH,
                                                                                pretrained_vec=pretrained_w2c)

        # 定义模型
        model = GRUBase(charge_label_size=len(lang.index2accu),
                        article_label_size=len(lang.index2art),
                        penalty_label_size=param.PENALTY_LABEL_SIZE,
                        pretrained_w2c=pretrained_w2c,
                        dropout=param.DROPOUT_RATE,
                        num_layers=param.GRU_LAYERS,
                        input_size=param.EM_SIZE,
                        hidden_size=param.GRU_HIDDEN_SIZE,
                        mode=param.MODE)
        model.to(device)
        # 定义损失函数
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW([{"params": model.em.parameters(), 'lr': 0.00001},
                            {"params": model.enc.parameters(), 'weight_decay': 0.05},
                            {'params': model.chargePreds.parameters()},
                            {'params': model.articlePreds.parameters()},
                            {'params': model.penaltyPreds.parameters()}
                            ], lr=param.LR, weight_decay=param.L2)

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
                                                                       num_warmup_steps=param.WARMUP_STEP,
                                                                       num_training_steps=param.STEP,
                                                                       num_cycles=param.NUM_CYCLES)
        print("training method ...\n")

        train_loss = 0
        train_loss_records = []
        valid_loss_records = []
        valid_acc_records = {"charge": [], "article": [], "penalty": []}
        valid_mp_records = {"charge": [], "article": [], "penalty": []}
        valid_f1_records = {"charge": [], "article": [], "penalty": []}
        valid_mr_records = {"charge": [], "article": [], "penalty": []}
        for epoch in range(param.EPOCH): # 50个epoch
            start = time.time()
            for seqs, accu_labels, article_labels, penalty_labels in utils.data_loader(train_seq,
                                                                               train_charge_labels,
                                                                               train_article_labels,
                                                                               train_penalty_labels,
                                                                               shuffle=True,
                                                                               batch_size=param.BATCH_SIZE):

                # 设置模型状态
                model.train()

                # 优化参数的梯度置0
                optimizer.zero_grad()

                seq_lens = [len(s) for s in seqs]
                for i in range(len(seqs)):
                    seqs[i] = torch.tensor(seqs[i])
                padded_input_ids = pad_sequence(seqs, batch_first=True).to(device)

                if param.MODE == "multi":
                    charge_preds, article_preds, penalty_preds = model(padded_input_ids, seq_lens)

                    # 指控分类误差
                    charge_preds_loss = criterion(charge_preds, torch.tensor(accu_labels).to(device))

                    # 法律条款预测误差
                    article_preds_loss = criterion(article_preds, torch.tensor(article_labels).to(device))

                    # 刑期预测误差
                    penalty_preds_loss = criterion(penalty_preds, torch.tensor(penalty_labels).to(device))

                    loss = (charge_preds_loss + article_preds_loss+ penalty_preds_loss)
                
                if param.MODE == "charge":
                    charge_preds = model(padded_input_ids, seq_lens)

                    # 指控分类误差
                    loss = criterion(charge_preds, torch.tensor(accu_labels).to(device))

                if param.MODE == "article":
                    article_preds = model(padded_input_ids, seq_lens)

                    # 法律条款预测误差
                    loss = criterion(article_preds, torch.tensor(article_labels).to(device))

                if param.MODE == "multi":
                    penalty_preds = model(padded_input_ids, seq_lens)

                    # 刑期预测误差
                    loss = criterion(penalty_preds, torch.tensor(penalty_labels).to(device))
                
                train_loss += loss.item()

                # 反向传播计算梯度
                loss.backward()

                # 梯度裁剪防止梯度爆炸
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # 更新梯度
                optimizer.step()

                # 更新学习率
                scheduler.step()

            train_loss_records.append(train_loss/len(train_seq))

            # 训练完一个EPOCH后评价模型
            # 初始化混淆矩阵
            if param.MODE == "multi" or param.MODE == "charge":
                charge_confusMat = utils.ConfusionMatrix(len(lang.index2accu))
            if param.MODE == "multi" or param.MODE == "article":
                article_confusMat = utils.ConfusionMatrix(len(lang.index2art))
            if param.MODE == "multi" or param.MODE == "penalty":
                penalty_confusMat = utils.ConfusionMatrix(param.PENALTY_LABEL_SIZE)
            
            # 验证模型在验证集上的表现
            model.eval()
            valid_loss = 0

            for val_seq, val_charge_label, val_article_label, val_penalty_label in utils.data_loader(test_seq,
                                                                                               test_charge_labels,
                                                                                               test_article_labels,
                                                                                               test_penalty_labels,
                                                                                               shuffle=False,
                                                                                               batch_size=5 * param.BATCH_SIZE):
                val_seq_lens = [len(s) for s in val_seq]
                val_input_ids = [torch.tensor(s) for s in val_seq]
                val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(device)
                with torch.no_grad():
                    if param.MODE == "multi":
                        val_charge_preds, val_article_preds, val_penalty_preds = model(val_input_ids, val_seq_lens)
                        val_charge_preds_loss = criterion(val_charge_preds, torch.tensor(val_charge_label).to(device))
                        val_article_preds_loss = criterion(val_article_preds, torch.tensor(val_article_label).to(device))
                        val_penalty_preds_loss = criterion(val_penalty_preds, torch.tensor(val_penalty_label).to(device))
                        valid_loss += (val_charge_preds_loss.item()+val_article_preds_loss.item()+val_penalty_preds_loss.item())

                    if param.MODE == "charge":
                        val_charge_preds = model(val_input_ids, val_seq_lens)
                        valid_loss = criterion(val_charge_preds, torch.tensor(val_charge_label).to(device))

                    if param.MODE == "article":
                        val_article_preds = model(val_input_ids, val_seq_lens)
                        valid_loss = criterion(val_article_preds, torch.tensor(val_article_label).to(device))
                        
                    if param.MODE == "penalty":
                        val_penalty_preds = model(val_input_ids, val_seq_lens)
                        valid_loss = criterion(val_penalty_preds, torch.tensor(val_penalty_label).to(device))
                if param.MODE == "multi" or param.MODE == "charge":        
                    charge_confusMat.updateMat(val_charge_preds.cpu().numpy(), np.array(val_charge_label))
                if param.MODE == "multi" or param.MODE == "article":  
                    article_confusMat.updateMat(val_article_preds.cpu().numpy(), np.array(val_article_label))
                if param.MODE == "multi" or param.MODE == "penalty":  
                    penalty_confusMat.updateMat(val_penalty_preds.cpu().numpy(), np.array(val_penalty_label))

            valid_loss_records.append(valid_loss/len(test_seq))

            # acc
            if param.MODE == "multi" or param.MODE == "charge":  
                valid_acc_records['charge'].append(charge_confusMat.get_acc())
            if param.MODE == "multi" or param.MODE == "article":  
                valid_acc_records['article'].append(article_confusMat.get_acc())
            if param.MODE == "multi" or param.MODE == "penalty":  
                valid_acc_records['penalty'].append(penalty_confusMat.get_acc())

            # F1
            if param.MODE == "multi" or param.MODE == "charge":  
                valid_f1_records['charge'].append(charge_confusMat.getMaF())
            if param.MODE == "multi" or param.MODE == "article":  
                valid_f1_records['article'].append(article_confusMat.getMaF())
            if param.MODE == "multi" or param.MODE == "penalty":  
                valid_f1_records['penalty'].append(penalty_confusMat.getMaF())

            # MR
            if param.MODE == "multi" or param.MODE == "charge":  
                valid_mr_records['charge'].append(charge_confusMat.getMaR())
            if param.MODE == "multi" or param.MODE == "article":  
                valid_mr_records['article'].append(article_confusMat.getMaR())
            if param.MODE == "multi" or param.MODE == "penalty":  
                valid_mr_records['penalty'].append(penalty_confusMat.getMaR())

            # MP
            if param.MODE == "multi" or param.MODE == "charge":  
                valid_mp_records['charge'].append(charge_confusMat.getMaP())
            if param.MODE == "multi" or param.MODE == "article":  
                valid_mp_records['article'].append(article_confusMat.getMaP())
            if param.MODE == "multi" or param.MODE == "penalty":  
                valid_mp_records['penalty'].append(penalty_confusMat.getMaP())

            #save model
            torch.save(model, f"output/train_gru/model-{lang.name}-{param.MODE}-{epoch}.pkl")

            print(f"Epoch: {epoch}  Train_loss: {round(train_loss / len(train_seq), 6)}  Valid_loss: {round(valid_loss/len(test_seq), 6)}")
            if param.MODE == "multi" or param.MODE == "article":
                print(f"Article_Acc: {round(article_confusMat.get_acc(), 6)}  Article_F1: {round(article_confusMat.getMaF(), 6)}  Article_MR: {round(article_confusMat.getMaR(), 6)}  Article_MP: {round(article_confusMat.getMaP(), 6)}")
            if param.MODE == "multi" or param.MODE == "charge":
                print(f"Charge_Acc: {round(charge_confusMat.get_acc(), 6)}  Charge_F1: {round(charge_confusMat.getMaF(), 6)}  Charge_MR: {round(charge_confusMat.getMaR(), 6)}  Charge_MP: {round(charge_confusMat.getMaP(), 6)}")
            if param.MODE == "multi" or param.MODE == "penalty":
                print(f"Penalty_Acc: {round(penalty_confusMat.get_acc(), 6)}  Penalty_F1: {round(penalty_confusMat.getMaF(), 6)}  Penalty_MR: {round(penalty_confusMat.getMaR(), 6)}  Penalty_MP: {round(penalty_confusMat.getMaP(), 6)}")
            print(f"Time: {round((time.time() - start) / 60, 2)}min")

train()

