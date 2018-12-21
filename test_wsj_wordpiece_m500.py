
from qrnn_wordpiece_ctc import Model_CTC_QRNN
import sentencepiece as spm

from wsj_data.dataset_wordpiece import WSJDataSet
from config import Config
import numpy as np

batch_size = 1
n_label = 31

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
base_path = '/home/jhpark/raw_wav/'
data_path = '../wsj_data/data/'

sp = spm.SentencePieceProcessor()
sp.Load('./m500_wsj.model')
dataset = WSJDataSet(batch_size, charset, base_path, sp, sample_rate = 2, data_path = data_path, reduced=False)

cfg = Config()
cfg.rnnSize = 1000
cfg.initial_lr = 3e-4
cfg.dropout = 0.75
filter_width = 1
n_layers = 6
mid_filter = 15
conv_filter = 5

model = Model_CTC_QRNN(label_size = n_label, vocabularySize=sp.GetPieceSize(), batch_size = batch_size, dropout=cfg.dropout, zoneout=0.0, rnnSize = cfg.rnnSize, clip = cfg.clip_norm, n_layers=n_layers, k_width=filter_width, mid_filter=mid_filter, conv_filter = conv_filter )

model_path = './model/qrnn_wordpiece_x2_x2_1000_6_data_all-99'
#model_path = './model/qrnn_wordpiece_x2_x2_700_6_new-134'

#model_path = '/raid/jhpark/model/qrnn/qrnn_conv_am_k_1_layer_6_350_mid_conv_15_not_shared-103'
#model_path = '/raid/jhpark/model/qrnn/qrnn_conv_am_k_1_layer_6_700_mid_conv_11_not_shared-108'
#model_path = '/data/model/qrnn/qrnn_conv_bias_am_k_1_layer_6_700_mid_conv_15_drop_only-35'
#model_path = './model/qrnn_conv_bias_ifo_am_k_1_layer_6_700_mid_conv_15_si284_restore-52'
#model_path = './model/sru2_am_5_632_lr3e3'
model.test(dataset, sp,  path=model_path) 


