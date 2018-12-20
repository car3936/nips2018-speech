import numpy as np
import pickle

from .preprocessing import *
from .utils import *

from scipy.io import wavfile


class WSJDataSet(object):
    """
    
    """

    def __init__(self, batch_size, charset, base_path,sample_rate=2, data_path = './data/', preprocessed = True, reduced = False):
        self.processed = preprocessed
        self.sample_rate = sample_rate
        if self.processed:
            with open(base_path+'train_si284_processed.list', 'r') as f:
                print('read trainlist')
                self.train_list = f.readlines()
        else:
            with open(base_path + 'train_all_wav.list', 'r') as f:
                self.train_list = f.readlines()

        with open(base_path + 'train_si284_processed.trans', 'r') as f:
            print('read trans')
            self.train_label = f.readlines()

        if self.processed:
            with open(base_path + 'test_dev93_processed.list', 'r') as f:
                self.valid_list = f.readlines()
        else:
            with open(base_path + 'test_dev93_wav.list', 'r') as f:
                self.valid_list = f.readlines()

        with open(base_path + 'test_dev93_wav.trans', 'r') as f:
            self.valid_label = f.readlines()

        if self.processed:
            with open(base_path + 'test_eval92_processed.list', 'r') as f:
                self.test_list = f.readlines()
        else:
            with open(base_path + 'test_eval92_wav.list', 'r') as f:
                self.test_list = f.readlines()

        with open(base_path + 'test_eval92_wav.trans', 'r') as f:
            self.test_label = f.readlines()


        train_idx_small = []
        train_idx_mid = []
        train_idx_big = []
        train_idx_all = []
        n_file = len(self.train_list) 


        self.debug_idx = 0
        #print('total file : ', n_file)
        for i in range(n_file):
            l = self.train_list[i]
            t = self.train_label[i]
            if self.processed:
                n_frame = np.load(base_path + l[:-1]).shape[0]
                #n_frame_compressed = np.ceil(n_frame/4).astype('int32')
                n_frame_compressed = np.ceil(n_frame/sample_rate).astype('int32')
            else:
                wav_path = base_path + l[:-1]
                _, sig = wavfile.read(wav_path)

                n_frame = 1 + np.floor((len(sig) - 400) / 160).astype('int32')
                #n_frame_compressed = np.ceil(n_frame/4).astype('int32')
                n_frame_compressed = np.ceil(n_frame/sample_rate).astype('int32')
            if (len(t) + 5) >= n_frame_compressed:                
                print(i+1,'th sentence err')
                continue

            if n_frame < 400 :
                train_idx_small.append(i)

            if n_frame < 800 :
                train_idx_mid.append(i)

            if n_frame < 1200 :
                train_idx_big.append(i)

            if n_frame < 1600 :
                train_idx_all.append(i)

        self.train_idx_under_400 = np.asarray(train_idx_small, dtype='int32')
        self.train_idx_under_800 = np.asarray(train_idx_mid, dtype='int32')
        self.train_idx_under_1200 = np.asarray(train_idx_big, dtype='int32')
        self.train_idx_under_1600 = np.asarray(train_idx_all, dtype='int32')
       
        if reduced:
            self.train_idx_under_400 = self.train_idx_under_400[::2]
            self.train_idx_under_800 = self.train_idx_under_800[::2]
            self.train_idx_under_1200 = self.train_idx_under_1200[::2]
            self.train_idx_under_1600 = self.train_idx_under_1600[::2]

        print('# of small dataset : ', self.train_idx_under_400.shape[0])
        print('# of mid dataset : ', self.train_idx_under_800.shape[0])
        print('# of big dataset : ', self.train_idx_under_1200.shape[0])
        print('# of all dataset : ', self.train_idx_under_1600.shape[0])


        if self.processed == False:
            self.mean = np.load(base_path + 'mean.npy').T
            self.var = np.load(base_path + 'var.npy').T
            self.std = np.sqrt(self.var)
            self.fb = np.load(data_path + 'fb.npy')

        self.base_path = base_path

        self.mode = '' #mode : train_under_400, train_under_800, train_under_1200, train_under_1600, train_all, valid, test

        self.counter = 0

        self.n_data = len(self.train_list)
        self.data_idx_perm = np.random.permutation(self.n_data)
        self.batch_size = batch_size
        self.n_batch = int(self.n_data / self.batch_size)

        #char <-> label
        self.charset = charset
        self.char_to_label = dict()
        self.label_to_char = dict()
        for i in range(len(self.charset)):
            self.char_to_label[self.charset[i]] = i
            self.label_to_char[i] = self.charset[i]





    def reset(self):
        self.counter = 0
        self.data_idx_perm = np.random.permutation(self.n_data)
        self.n_batch = int(self.n_data / self.batch_size)

    def set_mode(self, mode):
        self.mode = mode

        if self.mode == 'train_under_400':
            self.n_data = self.train_idx_under_400.shape[0]
        elif self.mode == 'train_under_800':
            self.n_data = self.train_idx_under_800.shape[0]
        elif self.mode == 'train_under_1200':
            self.n_data = self.train_idx_under_1200.shape[0]
        elif self.mode == 'train_under_1600':
            self.n_data = self.train_idx_under_1600.shape[0]
        elif self.mode == 'train_all':
            self.n_data = len(self.train_list)
        elif self.mode == 'valid':
            self.n_data = len(self.valid_list)
        elif self.mode == 'test':
            self.n_data = len(self.test_list)
        else:
            print('wrong data mode')
            raise NotImplementedError
        self.reset()

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.reset()

    def iter_flag(self):
        if self.counter < self.n_batch:
            return True
        else:
            return False

    def get_data(self):

        sample_rate = self.sample_rate
        data_idx = self.data_idx_perm[self.counter*self.batch_size : (self.counter+1) * self.batch_size]
        if self.mode == 'train_under_400':
            data_idx = self.train_idx_under_400[data_idx]
        elif self.mode == 'train_under_800':
            data_idx = self.train_idx_under_800[data_idx]
        elif self.mode == 'train_under_1200':
            data_idx = self.train_idx_under_1200[data_idx]
        elif self.mode == 'train_under_1600':
            data_idx = self.train_idx_under_1600[data_idx]
        elif self.mode == 'train_all':
            pass

        # no shuffle for valid and test
        elif self.mode == 'valid':
            data_idx = np.asarray(range(self.counter*self.batch_size, (self.counter+1)*self.batch_size), dtype = 'int32')
        elif self.mode == 'test':
            data_idx = np.asarray(range(self.counter * self.batch_size, (self.counter + 1) * self.batch_size),
                                  dtype='int32')
        else:
            print('wrong data mode')
            raise NotImplementedError

        # preprocessing data
        data_x = []
        data_y = []
        seq_len = []
        string_set = []
        #for debug
        #data_idx = [self.counter]
        #data_idx = range(self.counter*self.batch_size+64*700 + 300*10 +700, (self.counter+1) * self.batch_size + 64*700 + 300*10+700)
        #self.debug_idx = data_idx[0]
        for i in data_idx:
            if self.mode == 'valid':
                file_name = self.valid_list[i][:-1]
                target_string = self.valid_label[i]
            elif self.mode == 'test':
                file_name = self.test_list[i][:-1]
                target_string = self.test_label[i]
            else:
                file_name = self.train_list[i][:-1]
                target_string = self.train_label[i]


            if self.processed:
                #with open(self.base_path+file_name, 'rb') as f_pkl:
                #    x = pickle.load(f_pkl)
                #    print('file name : ', self.base_path + file_name)
                x = np.load(self.base_path + file_name)
            else:
                x = get_data_x(self.base_path, file_name, self.fb)
                x = (x-self.mean) / self.std

            string_set.append(target_string)
            label = string_to_label(target_string, self.char_to_label)
            seq_len.append(x.shape[0])
#            if len(label) > np.ceil(x.shape[0]/4).astype('int32') :
            if len(label) > np.ceil(np.ceil(x.shape[0]/sample_rate)).astype('int32') :
                print(i,'th training data label len : ', len(label), 'n_frame : ', np.ceil(x.shape[0]/sample_rate))
            data_y.append(label)
            data_x.append(x)

        seq_len = np.asarray(seq_len, dtype = 'int32')
        max_seq = np.max(seq_len)
        seq_len_compressed = np.ceil(seq_len / sample_rate).astype('int32')
        #seq_len_compressed = np.ceil(seq_len / 2).astype('int32')
        max_seq_compressed = np.max(seq_len_compressed)

        #
        x = np.zeros((self.batch_size, max_seq, 40, 3), dtype='float32')
        for i in range(self.batch_size):
            x[i,0:seq_len[i], :, :] = data_x[i]


        sparse_indices, sparse_values, sparse_shape = list_to_sparse_tensor(data_y, max_seq_compressed)
        self.counter += 1

        return x, seq_len_compressed, sparse_indices, sparse_values, sparse_shape, string_set



if __name__ == '__main__':
    batch_size = 30
    base_path = '/home/yhboo/skt/wavectrl/raw_wav/'
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ .'\n"
    dataset = WSJDataSet(batch_size, charset, base_path, data_path='./data/')
    dataset.set_mode('train_under_800')
    print(dataset.n_data)
    print(dataset.n_batch)
    print(len(dataset.train_list))
    counter = 0
    exit(0)
    while(dataset.iter_flag()):
    #while True:
        x, seq_len, sparse_indices, sparse_values, sparse_shape,_ = dataset.get_data()
        print('data shape : ', x.shape)
        #print('seq len :', seq_len)
        #print('sparse indices : ', sparse_indices.shape)
        #print('sparse values : ', sparse_values.shape)
        #print('sparse shape : ', sparse_shape)
        print('max indices :', np.max(sparse_indices, axis = 0))
        if x.shape[1] > 800:
            break



    print('n_batch : ', counter)




