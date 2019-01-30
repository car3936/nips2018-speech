from models.lstm_wordpiece_ctc import Model_CTC_LSTM
from wsj_data.dataset_wordpiece import WSJDataSet
from config import Config
import numpy as np
import sentencepiece as spm

batch_size = 50
n_label = 31

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
base_path = '/home/jhpark/raw_wav/'
data_path = './wsj_data/data/' # statistics 
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

model_path = './saved_models/lstm_wordpiece_x2_x2_600_4'
train_curve, valid_curve, cer_curve = model.train(dataset, cfg.initial_lr, cfg.max_patience , cfg.max_change, batch_size=batch_size,charset=charset, sp=sp, max_epoch= cfg.max_epoch, path=model_path, regularizer=0.01) 

np.save(model_path + '_train_curve',np.array(train_curve))
np.save(model_path + '_valid_curve',np.array(valid_curve))
np.save(model_path + '_cer_curve',np.array(cer_curve))

