import sys
sys.path.insert(0,"/home/jhpark/tf_projects/")

from wsj_data.dataset_gating import WSJDataSet
import numpy as np

batch_size = 32
n_ptb_char = 50
n_label = 31

charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
base_path = '/home/yhboo/skt/wavectrl/raw_wav/'
data_path = '/home/jhpark/tf_projects/wsj_data/data/'
boundary_path = '/home/jhpark/segmentation_wsj/'

dataset = WSJDataSet(batch_size, charset, base_path, sample_rate = 2, data_path = data_path, boundary_path = boundary_path)
dataset.set_mode('train_under_1600')

while dataset.iter_flag():
    batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape, gate,  _ = dataset.get_data()
    print(np.shape(batch_x))
    print(np.shape(sparse_indices))
    print(np.shape(sparse_values))
    print(np.shape(sparse_shape))
    print(np.shape(gate))
    print((sparse_shape))
    break


