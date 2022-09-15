#coding:utf-8
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import configparser
import utils
from models import GRUBase
import torch


def train():
    print("load model params...")
    param = utils.Params("gru-base")

    model = GRUBase()



