import os, sys
# sys.path.append(os.path.dirname())
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import gensim
import numpy as np
import utils
from torch.nn.utils.rnn import pad_sequence
from utils import prepare_data, data_loader, Lang

# 分析gru输出是否可以在欧式空间区分开similarity
def ana_gru_output():
    BS = 15
    corpus_info_path = "dataset/CAIL-SMALL/lang.pkl"
    CPATH = "dataset/case_study.txt"
    MPATH = [f"output/train_gru/model-small-charge-2.pkl"]
    w2c_Path = f'dataset/pretrained_w2c/law_token_vec_300.bin'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("load model params...")
    param = utils.Params("gru-base")

    print("loading corpus info")
    f = open(corpus_info_path, "rb")
    lang = pickle.load(f)
    f.close()

    print("loading pretrained word2vec")
    pretrained_w2v = gensim.models.KeyedVectors.load_word2vec_format(w2c_Path, binary=False)

    arr = torch.zeros(size=(BS,BS))
    for k in range(len(MPATH)):
        print("loading model...")
        model = torch.load(MPATH[k], map_location=torch.device('cpu'))
        model.to(device)
        model.eval()

        print(f"loading cases...")
        seq, charge_labels, article_labels, penalty_labels = prepare_data("dataset/CAIL-SMALL/case_study.txt", 
                                                                            lang,
                                                                            max_length=param.MAX_LENGTH, 
                                                                            pretrained_vec=pretrained_w2v)

        for val_seq, val_charge_label, val_article_label, val_penalty_label in data_loader(seq,
                                                                                            charge_labels,
                                                                                            article_labels,
                                                                                            penalty_labels,
                                                                                            shuffle = False,
                                                                                            batch_size=BS):
            val_seq_lens = [len(s) for s in val_seq]
            val_input_ids = [torch.tensor(s) for s in val_seq]
            val_input_ids = pad_sequence(val_input_ids, batch_first=True).to(device)
            with torch.no_grad():
                dist = nn.PairwiseDistance(p=2)
                val_charge_vecs = model(val_input_ids, val_seq_lens)
                # charge_vecs = val_charge_vecs.tolist()
                for i in range(BS):
                    for j in range(BS):
                        arr[i][j] = round(dist(val_charge_vecs[i], val_charge_vecs[j]).item(), 2)
        print(arr.numpy())

        v_image = plt.imshow(arr.numpy(), cmap='viridis')
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        plt.colorbar(v_image, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(0, BS, 1))#不显示坐标刻度
        plt.yticks(np.arange(0, BS, 1))
        plt.savefig(f"output/analysis/gru/case_study-{k}.pdf",bbox_inches='tight')
        plt.show()

    

def ana_gru_lsscl_output():
    # 分析gru+lsscl输出是否可以在欧式空间区分开confusing law
    pass


if __name__=="__main__":
    ana_gru_output()