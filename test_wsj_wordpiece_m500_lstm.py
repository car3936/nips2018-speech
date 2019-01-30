from models.lstm_wordpiece_ctc import Model_CTC_LSTM
import sentencepiece as spm

from wsj_data.dataset_wordpiece import WSJDataSet
from config import Config
import numpy as np

batch_size = 1
n_label = 31

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
base_path = '/home/jhpark/raw_wav/'
data_path = './wsj_data/data/'

sp = spm.SentencePieceProcessor()
sp.Load('./m500_wsj.model')
dataset = WSJDataSet(batch_size, charset, base_path, sp, sample_rate = 2, data_path = data_path, reduced=False)

cfg = Config()

cfg.rnnSize = 600
cfg.initial_lr = 3e-4
cfg.dropout = 0.75
filter_width = 1
n_layers = 4
mid_filter = 15
conv_filter = 5

model = Model_CTC_LSTM(label_size = n_label, vocabularySize=sp.GetPieceSize(), batch_size = batch_size, dropout=cfg.dropout, zoneout=0.0, rnnSize = cfg.rnnSize, clip = cfg.clip_norm, n_layers=n_layers, k_width=filter_width, mid_filter=mid_filter, conv_filter = conv_filter )

model_path = './saved_models/lstm_wordpiece_600_4'
model.test(dataset, sp,  path=model_path) 


