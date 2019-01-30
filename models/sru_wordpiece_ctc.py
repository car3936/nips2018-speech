
import tensorflow as tf
import time
import numpy as np
import pickle
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.layers import xavier_initializer 
from tensorflow.python import debug as tf_debug

from tensorflow.contrib.rnn import LSTMStateTuple
from sru_layer import SRU_layer
import sentencepiece as spm

def CER(label, predict):
    """
    x, y : string for one sentence
    x : label
    y : predict
    output : wer
    """

    x = label
    y = predict
    d = np.zeros(((len(x) + 1), (len(y) + 1)), dtype='float32')

    for i in range(len(x) + 1):
        for j in range(len(y) + 1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitute = d[i - 1][j - 1] + 1
                insert = d[i][j - 1] + 1
                delete = d[i - 1][j] + 1
                d[i][j] = min(substitute, insert, delete)
    result = float((d[len(x)][len(y)]) / len(x))

    return result
class Model_CTC_SRU(object):

    def __init__(self, stochastic = False, use_slope = True, variational_dropout = False, vocabularySize=283, label_size=50,  rnnSize=256, n_layers=3, dropout=0.5, zoneout=0.1, embedding_size=None, dtype=tf.float32, clip = 0.35, k_width=3, name = 'hlstm', conv_filter=3, mid_filter = 25,  batch_size = 128 ):
        self.rnnSize = rnnSize
        self.vocabularySize = vocabularySize
        self.outputSize = label_size 
        self.stochastic = stochastic

        self.dtype = dtype
        self.dropout = dropout
        self.n_layers = n_layers
        self.clip = clip
        self.name = name
        self.use_slope = use_slope
        self.zoneout = zoneout
        self.batch_size = batch_size
        self.k_width = k_width
        self.conv_filter = conv_filter 

        self.mid_filter = mid_filter
        f_bias = 0.0
       
        # placeholders 
        self.x = tf.placeholder(tf.float32, [None, None, 40, 3],name = 'x') #[batch, seq_len]
        self.label = tf.sparse_placeholder(tf.int32, name = 'label') #[batch, seq_len]
        self.seq_len = tf.placeholder(tf.int32, [None], name = 'seq_len')        # [batch_size]
        
        self.is_train = tf.placeholder(tf.bool, [], name='train')       
        self.lr = tf.placeholder(tf.float32, [], name='lr')
        dropout_p = tf.where(self.is_train, self.dropout, 1.0) 
        dropout_p = tf.cast(dropout_p, dtype=self.dtype)


        # LSTM layers 
        self.lstm_cells = [] 
        conv_filter_size = (self.conv_filter, self.conv_filter)
        
        h = tf.layers.conv2d(self.x, 32, conv_filter_size, (2,2), 'same', use_bias= False, name = 'conv0')
        h = tf.contrib.layers.batch_norm(h, center = True, scale = True, is_training=self.is_train, decay=0.9, epsilon=1e-3, scope='bn0')
        h = tf.nn.tanh(h, name='tanh0')

        h = tf.layers.conv2d(h, 32, conv_filter_size, (1, 2), 'same', use_bias=False, name = 'conv1')
        h = tf.contrib.layers.batch_norm(h, center=True, scale=True, is_training=self.is_train, decay=0.9, epsilon=1e-3, scope='bn1')
        h = tf.nn.tanh(h, name='tanh1')
        time_convolution = 2
        _seq_len_char = self.seq_len
        _seq_len_word = tf.div(self.seq_len, 2)
        #reshape
        # ([0] : batch_size, [1] : seq_len, [2]*[3] : feature dimension)
        h_shape = tf.shape(h)
        h = tf.reshape(h, [batch_size, h_shape[1], 1, 320])
        conv_filter = tf.get_variable('SRU_conv0_filter', shape = [mid_filter,1, 320, 1], trainable=True)
        h = tf.nn.depthwise_conv2d(h, conv_filter, [1,1,1,1], padding='SAME',  name = 'QRNN_conv0')
        h = tf.squeeze(h, axis = [-2])
        sru_ = SRU_layer(self.rnnSize, batch_size=self.batch_size, fwidth=self.k_width, pool_type = 'ifo', zoneout=self.zoneout, name='QRNN_layer0', infer = tf.logical_not(self.is_train),skip = True, skip_embedding = True)
        sru_h, last_state = sru_(h)
        sru_h = tf.nn.dropout(sru_h, dropout_p, noise_shape = [tf.shape(sru_h)[0],1,tf.shape(sru_h)[2]]) 

        for i in range(1,self.n_layers ):
            sru_h = tf.expand_dims(sru_h, -2)
            conv_filter = tf.get_variable('SRU_conv{}_filter'.format(i), shape = [mid_filter,1, self.rnnSize, 1], trainable=True)
            sru_h = tf.nn.depthwise_conv2d(sru_h, conv_filter, [1,1,1,1], padding='SAME',  name = 'QRNN_conv{}'.format(i))
            sru_h = tf.squeeze(sru_h, axis = [-2])
            sru_ = SRU_layer(self.rnnSize, batch_size=self.batch_size, fwidth=self.k_width, pool_type = 'ifo', zoneout=self.zoneout, name='QRNN_layer{}'.format(i), infer = tf.logical_not(self.is_train),skip = True, skip_embedding = False)
            print(sru_h)
            sru_h, last_state = sru_(sru_h)
            sru_h = tf.nn.dropout(sru_h, dropout_p, noise_shape = [tf.shape(sru_h)[0],1,tf.shape(sru_h)[2]]) 
            if i == self.n_layers - 3:
                character_h = sru_h
                sru_h = tf.expand_dims(sru_h, -2)
                conv_filter = tf.get_variable('SRU_time_conv_filter', shape = [time_convolution, 1, self.rnnSize, 1], trainable=True)
                sru_h = tf.nn.depthwise_conv2d(sru_h, conv_filter, [1, time_convolution, time_convolution,1], padding='SAME',  name = 'SRU_time_conv')
                sru_h = tf.squeeze(sru_h, axis = [-2])

        # character-ctc layer
        h_shape = tf.shape(character_h)
        output_h = tf.reshape(character_h, [-1, self.rnnSize])
        print(output_h)

        with tf.variable_scope('dense_character'): 
            dense = tf.layers.dense(output_h, self.outputSize, kernel_initializer=tf.random_uniform_initializer(-0.1,0.1))

        self.char_logit = tf.reshape(dense, [h_shape[0], h_shape[1], self.outputSize])
        self.char_loss = tf.nn.ctc_loss(
                inputs = self.char_logit,
                labels = self.label,
                sequence_length= _seq_len_char,
                time_major = False
            )
        self.char_loss =tf.reduce_mean(self.char_loss)
        train_loss = self.char_loss 
        char_opt = tf.train.AdamOptimizer(self.lr) 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad, var = zip(*char_opt.compute_gradients(train_loss))        
            clipped_gradients, _ = tf.clip_by_global_norm(grad,clip)
            self.char_optimizer = char_opt.apply_gradients(zip(clipped_gradients, var))

        self.sentence, _ = tf.nn.ctc_greedy_decoder(tf.transpose(self.char_logit, (1,0,2)), _seq_len_char)
        self.cer = tf.reduce_mean(tf.edit_distance(tf.cast(self.sentence[0], tf.int32),self.label))

        # wordpiece-ctc layer 
        h_shape = tf.shape(sru_h)
        output_h = tf.reshape(sru_h, [-1, self.rnnSize])
        print(output_h)

        with tf.variable_scope('dense_wordpiece'): 
            dense = tf.layers.dense(output_h, self.vocabularySize + 1, kernel_initializer=tf.random_uniform_initializer(-0.1,0.1))

        self.word_logit = tf.reshape(dense, [h_shape[0], h_shape[1], self.vocabularySize + 1])
        self.word_loss = tf.nn.ctc_loss(
                inputs = self.word_logit,
                labels = self.label,
                sequence_length= _seq_len_word,
                time_major = False
            )
        self.word_loss =tf.reduce_mean(self.word_loss)
        train_loss = self.word_loss 
        word_opt = tf.train.AdamOptimizer(self.lr) 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad, var = zip(*word_opt.compute_gradients(train_loss))        
            clipped_gradients, _ = tf.clip_by_global_norm(grad,clip)
            self.word_optimizer = word_opt.apply_gradients(zip(clipped_gradients, var))

        self.word_sentence, _ = tf.nn.ctc_greedy_decoder(tf.transpose(self.word_logit, (1,0,2)), _seq_len_word)
        print(self.word_sentence)
        self.word_sentence = tf.sparse_tensor_to_dense(self.word_sentence[0],default_value = 2 )
        #self.word_sentence = tf.sparse_tensor_to_dense(self.word_sentence.indices, self.word_sentence.shape, self.word_sentence.values,default_value = 2 )
        #self.word_distance = tf.reduce_mean(tf.edit_distance(tf.cast(word_sentence[0], tf.int32),self.wp_label))
        # last states to placeholder
        self.logsoftmax = tf.nn.log_softmax(self.word_logit)

        self.saver = tf.train.Saver()


    def greedy_decoding(self, predict, charset):
        predict = np.reshape(predict, (-1,))
        T = predict.shape[0]
        s = ""
        idx_blank = 30
        idx_eos = 29
        l = [predict[0]]

        #remove conjecutive labels
        for t in range(1,T):
            if predict[t] == l[-1]:
                pass
            else:
                l.append(predict[t])

        for t in range(len(l)):
            if l[t] == idx_blank:
                pass
            elif l[t] == idx_eos:
                s = s + ' <\s>'

            else:
                s = s + charset[l[t]]

        return s

    def load(self, path =''):
        sess = tf.Session()
        self.saver.restore(sess, save_path = path)

    def test(self, dataset, sp,  path='./model/hlstm'):
        loss = self.word_loss
        logit = self.word_logit
        test_loss = 0
        test_cer = 0
        test_cer2 = 0
        total_c = 0 
        sess = tf.Session()
        self.saver.restore(sess, save_path = path)
#        p_dict = dict()
#        for pp in tf.global_variables():
#            p_dict[pp.name] = sess.run(pp)
#        with open('./quant_weights_sru_wordpiece_all/float_params.pkl', 'wb') as f:
#            pickle.dump(p_dict, f)
#        exit()
        st_time = time.time()
        dataset.set_mode('test')
        idx = 0
        logit_list = []
        len_list = []
        logprob_list=[]
        trans_file = open(path + '_test_trans_fixed_all.txt','w+')
        result_file = open(path + '_test_result_fixed_all.txt','w+')

        while dataset.iter_flag():
            if idx % 100 == 0 and idx != 0:

                end_time = time.time()
                print('...... {:6d}, Time: {:.2f}, Loss: {:.4f}'.format(idx, end_time-st_time, test_loss/idx))
                st_time = time.time()
#                    break
            idx += 1
            batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape,  string_set = dataset.get_data()
            ops = [loss, logit,  self.word_sentence, self.logsoftmax]
            feed_dict = {
                    self.x: batch_x,
                    self.label: (sparse_indices, sparse_values, sparse_shape),
                    self.lr: 0.0,
                    self.seq_len: batch_seq_len,
                    self.is_train: False
                    }
            _loss, _logit, _sentence,  _logsoftmax  = sess.run(ops, feed_dict)
            test_loss += _loss
            logit_list.append(_logit)
   
            logprob_list.append(np.squeeze(_logsoftmax,axis = 0))
            len_list.append(_logit.shape[1])
            label_sentence = string_set
            _cer = 0
            decoded_sentence = []
            for _idx, s in enumerate(_sentence):
                decoded_sentence.append(sp.DecodeIds(s.tolist()))
                _cer += CER(label_sentence[_idx][:-1], decoded_sentence[_idx][:-2])
                trans_file.write(label_sentence[_idx][:-1] + '\n'  )
                result_file.write(decoded_sentence[_idx][:-2] + '\n')
                   
            _cer /= len(label_sentence)
                #print(_cer)
            test_cer += _cer 
            test_cer2 += _cer*_logit.shape[1]
            total_c += _logit.shape[1]

        test_loss /= dataset.n_batch
        test_cer /= dataset.n_batch
        test_cer2 /= total_c
        print('...... Test BPC', test_loss )
        #print('...... Test CER', test_cer )
        print('...... Test CER', test_cer2 )

        result_dict = dict()
        result_dict['logit'] = logit_list
        result_dict['logprob'] = logprob_list
        result_dict['length'] = len_list
        with open(path + '_test_logit.pkl', 'wb') as f:
            pickle.dump(result_dict, f)
        

    def train(self, dataset, lr, max_patience, max_change, batch_size, charset, sp, regularizer=0.0, max_epoch=200, path='./model/hlstm'):
#        optim = self.optimizer 
        char_optim = self.char_optimizer
        word_optim = self.word_optimizer
        char_loss = self.char_loss
        word_loss = self.word_loss
        cer = self.cer
        char_logit = self.char_logit
        word_logit = self.word_logit
        train_curve = []
        valid_curve = []
        g_norm_curve = []
        cer_curve = []

        dataset.set_mode('train_under_1600')
        '''train'''

        best_valid_loss = 1000000
        patience = 0
        change = 0
        status = 'keep_train'
        sess = tf.Session()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)

        sess.run(tf.global_variables_initializer())
        
        for epoch in range(10):

            train_loss = 0 
            valid_loss = 0
            train_cer = 0
            valid_cer = 0

            st_time = time.time()
            idx = 0
            dataset.set_mode('train_under_1600')
            while dataset.iter_flag():
                if idx % 100 == 0 and idx != 0:
                    end_time = time.time()
                    print('...... {:6d}/{:6d} , Time: {:.2f}, Loss: {:.4f}'.format(idx, dataset.n_batch, end_time-st_time, train_loss/idx))
                    st_time = time.time()
                idx +=1
                ops = [char_optim, char_loss, char_logit, cer]
                batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape,  _ = dataset.get_data(False)
                feed_dict = {
                    self.lr: lr,
                    self.x: batch_x,
                    self.label: (sparse_indices, sparse_values, sparse_shape),
                    self.seq_len: batch_seq_len,
                    self.is_train: True
                }
                _, _loss, _logit, _cer = sess.run(ops, feed_dict)

                train_loss += _loss
                train_cer += _cer
                if np.isnan(_loss):
                    break
            idx -= 1
            train_loss /= dataset.n_batch
            train_cer /= dataset.n_batch

            print('...... Train loss', train_loss)
            print('...... Train CER', train_cer)
        cur_lr = lr
        for epoch in range(max_epoch):
            
            print('... Epoch', epoch, status)

            start_time = time.time()

            # lr scheduling
            if status == 'end_train':
                time.sleep(1)
                self.saver.save(sess, save_path = path,global_step = epoch)
                best_epoch = epoch 
                break
            elif status == 'change_lr':
                time.sleep(1)
                self.saver.restore(sess, save_path = path + '-' + str(best_epoch))
                
                cur_lr = lr * np.power(0.2, change)
            elif status == 'save_param':
                self.saver.save(sess, save_path = path,global_step = epoch)
                best_epoch = epoch
            elif status == 'roll_back':
                self.saver.restore(sess, save_path = path + '-' + str(best_epoch))
            else:
                pass

            st_time = time.time()
            train_loss = 0 
            valid_loss = 0
            train_cer = 0
            valid_cer = 0
            train_g_norm = 0
            idx = 0
            ''' train wordpiece '''
            dataset.set_mode('train_under_1600')
            while dataset.iter_flag():
                if idx % 100 == 0 and idx != 0:
                    end_time = time.time()
                    print('...... {:6d} / {:6d}, Time: {:.2f}, Loss: {:.4f}'.format(idx, dataset.n_batch, end_time-st_time, train_loss/idx))
                    st_time = time.time()
                idx +=1
                batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape,  _ = dataset.get_data(True)
                
                ops = [word_optim, word_loss, word_logit]
                feed_dict = {
                        self.x: batch_x,
                        self.label: (sparse_indices, sparse_values, sparse_shape),
                        self.lr: cur_lr,
                        self.seq_len: batch_seq_len,
                        self.is_train: True
                        }
                _, _loss, _logit, = sess.run(ops, feed_dict)

                train_loss += _loss
#                train_cer += _cer
                if np.isnan(_loss):
                    break

            if np.isnan(train_loss):
                status = 'roll_back'
                epoch -= 1
                print('nan loss detected, roll back epoch')
                continue
            train_loss /= dataset.n_batch
#            train_cer /= dataset.n_batch
            train_curve.append(train_loss)
            print('...... Train loss', train_loss)
#            print('...... Train CER', train_cer)

            idx = 0
            dataset.set_mode('valid')
            while dataset.iter_flag():
                if idx % 100 == 0 and idx != 0:
                    end_time = time.time()
                    print('...... {:6d} / {:6d}, Time: {:.2f}, Loss: {:.4f}'.format(idx, dataset.n_batch, end_time-st_time, valid_loss/idx))
                    st_time = time.time()

                idx += 1

                batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape,  string_set = dataset.get_data(True)
                #ops = [loss, logit, cer]
                ops = [word_loss, word_logit, self.word_sentence]
                feed_dict = {
                        self.x: batch_x,
                        self.label: (sparse_indices, sparse_values, sparse_shape),
                        self.lr: cur_lr,
                        self.seq_len: batch_seq_len,
                        self.is_train: False
                        }
                _loss, _logit, _sentence  = sess.run(ops, feed_dict)
                valid_loss += _loss
                _cer = 0
                decoded_sentence = []
                label_sentence = string_set
                for _idx, s in enumerate(_sentence):
                    decoded_sentence.append(sp.DecodeIds(s.tolist()))
                    _cer += CER(label_sentence[_idx], decoded_sentence[_idx])
                _cer /= len(decoded_sentence)
                valid_cer += _cer 
                
                if np.isnan(_loss):
                    break
            if np.isnan(valid_loss):
                status = 'roll_back'
                epoch -= 1
                print('nan loss detected, roll back epoch')
                continue
            valid_loss /= dataset.n_batch
            valid_cer /= dataset.n_batch
            valid_curve.append(valid_loss)
            cer_curve.append(valid_cer)
            print('...... Valid loss, Valid CER, best CER', valid_loss, valid_cer, best_valid_loss)
            #_logit = np.argmax(_logit, axis = -1)
            #print(self.greedy_decoding(_logit, charset))
            for i in range(5):
                print(label_sentence[i])
                print(decoded_sentence[i])

            if valid_cer > best_valid_loss:
                patience += 1
                print('......... Current patience', patience)
                if patience >= max_patience:
                    change += 1
                    patience = 0
                    print('......... Current lr change', change)
                    if change >= max_change:
                        status = 'end_train'  # (load param, stop training)
                    else:
                        status = 'change_lr'  # (load param, change learning rate)
                else:
                    status = 'keep_train'  # (keep training)
            else:
                best_valid_loss = valid_cer
                patience = 0
                print('......... Current patience', patience)
                status = 'save_param'  # (save param, keep training)

            end_time = time.time()
            print('...... Time:', end_time - start_time)

            np.save(path + '_train_curve',np.array(train_curve))
            np.save(path + '_valid_curve',np.array(valid_curve))
            np.save(path + '_cer_curve',np.array(cer_curve))


        return train_curve, valid_curve, cer_curve    





