from models.lstm_ctc import Model_CTC_LSTM

from wsj_data.dataset_si284 import WSJDataSet
from config import Config
import numpy as np

batch_size = 1
n_label = 31

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
base_path = '/home/jhpark/raw_wav/'
data_path = './wsj_data/data/' # statistics 
dataset = WSJDataSet(batch_size, charset, base_path, sample_rate = 2, data_path = data_path)


cfg = Config()
cfg.rnnSize = 600
cfg.initial_lr = 3e-4
n_layers = 3

model = Model_CTC_LSTM(label_size = n_label, rnnSize = cfg.rnnSize, clip = cfg.clip_norm, n_layers=n_layers, stochastic = True )

model_path = './saved_models/lstm_am_4x600'
model.test(dataset, path=model_path) 


