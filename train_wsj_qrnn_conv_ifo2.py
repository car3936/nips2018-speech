import sys
sys.path.insert(0,"/home/jhpark/tf_projects/")

from qrnn_conv_bias_ifo_ctc import Model_CTC_QRNN

from wsj_data.dataset import WSJDataSet
from config import Config
import numpy as np

batch_size = 32
n_label = 31

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
base_path = '/home/jhpark/raw_wav/'
data_path = '/home/jhpark/tf_projects/wsj_data/data/'
dataset = WSJDataSet(batch_size, charset, base_path, sample_rate = 2, data_path = data_path, reduced=False)

cfg = Config()
cfg.rnnSize = 1000
cfg.initial_lr = 3e-4
filter_width = 1
n_layers = 6
mid_filter = 15
conv_filter = 5

model = Model_CTC_QRNN(label_size = n_label, batch_size = batch_size, dropout=cfg.dropout, zoneout=0.0, rnnSize = cfg.rnnSize, clip = cfg.clip_norm, n_layers=n_layers, k_width=filter_width, mid_filter=mid_filter, conv_filter = conv_filter )

model_path = './model/qrnn_conv_bias_ifo_am_k_1_layer_6_1000_mid_conv_15_siall'
#model_path = './model/sru2_am_5_632_lr3e3'
train_curve, valid_curve, cer_curve = model.train(dataset, cfg.initial_lr, cfg.max_patience , cfg.max_change, batch_size=batch_size,charset=charset,max_epoch= cfg.max_epoch, path=model_path, regularizer=0.01) 

np.save(model_path + '_train_curve',np.array(train_curve))
np.save(model_path + '_valid_curve',np.array(valid_curve))
np.save(model_path + '_cer_curve',np.array(cer_curve))

