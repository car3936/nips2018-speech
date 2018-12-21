
from sru_conv_bias_ifo_ctc import Model_CTC_SRU

from wsj_data.dataset_si284 import WSJDataSet
from config import Config
import numpy as np

batch_size = 1
n_label = 31

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
base_path = '/home/jhpark/raw_wav/'
data_path = './wsj_data/data/' # statistics 
dataset = WSJDataSet(batch_size, charset, base_path, sample_rate = 2, data_path = data_path, reduced=False)

cfg = Config()
cfg.rnnSize = 700
cfg.initial_lr = 3e-4
filter_width = 1
n_layers = 6
conv_filter = 5
mid_filter = 15

model = Model_CTC_SRU(label_size = n_label, batch_size = batch_size, rnnSize = cfg.rnnSize, clip = cfg.clip_norm, n_layers=n_layers, k_width=filter_width, conv_filter = conv_filter, mid_filter=mid_filter)

model_path = './model/sru_conv_ifo_6_700_si284'

model.test(dataset,  path=model_path) 


