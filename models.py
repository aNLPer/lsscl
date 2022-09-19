import torch
import pickle
import torch.nn as nn 
import numpy as np
import label_sim_graph_construction.graph_construction as graph_construction
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GRUBase(nn.Module):
    def __init__(self,
                 charge_label_size,
                 article_label_size,
                 penalty_label_size,
                 voc_size, # 词汇表
                 pretrained_model,
                 hidden_size=300, # 隐藏状态size，
                 num_layers=2,
                 bidirectional=True,
                 dropout=0.6,
                 mode="concat"):
        super(GRUBase, self).__init__()

        self.voc_size = voc_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.charge_label_size = charge_label_size
        self.article_label_size = article_label_size
        self.penalty_label_size = penalty_label_size
        self.mode = mode
        self.pretrained_model = pretrained_model

        # self.em = nn.Embedding(self.voc_size, self.hidden_size, padding_idx=0)
        vectors = torch.tensor(self.pretrained_model.vectors, dtype=torch.float32).to(device)
        self.em = nn.Embedding.from_pretrained(vectors, freeze=False)

        self.enc = nn.GRU(input_size=self.hidden_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout,
                          batch_first=True,
                          bidirectional=self.bidirectional)

        self.chargePreds = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.charge_label_size)
        )

        self.articlePreds = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.article_label_size),
        )

        self.penaltyPreds = nn.Sequential(
            nn.Linear(2*self.hidden_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.penalty_label_size),
        )

    def forward(self, input_ids, seq_lens):
        # [batch_size, seq_length, hidden_size]
        inputs = self.em(input_ids)

        inputs_packed = pack_padded_sequence(inputs, seq_lens, batch_first=True, enforce_sorted=False)
        # packed_output
        outputs_packed, h_n = self.enc(inputs_packed)

        # [batch_size, seq_len, 2*hidden_size]
        outputs_unpacked, unpacked_lens = pad_packed_sequence(outputs_packed, batch_first=True)
        # [batch_size, 2*hidden_size]
        outputs_sum = outputs_unpacked.sum(dim=1)
        unpacked_lens = unpacked_lens.unsqueeze(dim=1).to(device)
        outputs_mean = outputs_sum/unpacked_lens

        # [batch_size, charge_label_size]
        charge_preds = self.chargePreds(outputs_mean)
        # [batch_size, article_label_size]
        article_preds = self.articlePreds(outputs_mean)
        # [batch_size, penalty_label_size]
        penalty_preds = self.penaltyPreds(outputs_mean)

        return charge_preds, article_preds, penalty_preds
b = [[1],[2],[3],[4]]
a = np.random.choice(range(len(b)), size=2, replace=False)
print(a)
print([b[i] for i in a])