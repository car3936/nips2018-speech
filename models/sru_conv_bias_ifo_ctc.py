import tensorflow as tf
import time
import numpy as np
import pickle
from tensorflow.contrib.rnn import LSTMCell, MultiRNNCell, DropoutWrapper
from tensorflow.contrib.layers import xavier_initializer 
from tensorflow.python import debug as tf_debug

from tensorflow.contrib.rnn import LSTMStateTuple
from sru_layer import SRU_layer

class Model_CTC_SRU(object):

    def __init__(self, stochastic = False, use_slope = True, variational_dropout = False, vocabulary_size=283, label_size=50,  rnnSize=256, n_layers=3, dropout=0.5, zoneout=0.1, embedding_size=None, dtype=tf.float32, clip = 0.35, k_width=3, name = 'hlstm', conv_filter=3, mid_filter = 25,  batch_size = 128):
        self.rnnSize = rnnSize
        self.inputSize = vocabulary_size
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

        #reshape
        # ([0] : batch_size, [1] : seq_len, [2]*[3] : feature dimension)
        h_shape = tf.shape(h)

        h = tf.reshape(h, [batch_size, h_shape[1], 1, 320])
        conv_filter = tf.get_variable('QRNN_conv0_filter', shape = [mid_filter,1, 320, 1], trainable=True)
        h = tf.nn.depthwise_conv2d(h, conv_filter, [1,1,1,1], padding='SAME',  name = 'QRNN_conv0')
        h = tf.squeeze(h, axis = [-2])
        sru_ = SRU_layer(self.rnnSize, batch_size=self.batch_size, fwidth=self.k_width, pool_type = 'ifo', zoneout=self.zoneout, name='QRNN_layer0', infer = tf.logical_not(self.is_train),skip = True, skip_embedding = True)
        sru_h, last_state = sru_(h)
        sru_h = tf.nn.dropout(sru_h, dropout_p, noise_shape = [tf.shape(sru_h)[0],1,tf.shape(sru_h)[2]])
       

        for i in range(1,self.n_layers ):
            sru_h = tf.expand_dims(sru_h, -2)
            conv_filter = tf.get_variable('QRNN_conv{}_filter'.format(i), shape = [mid_filter,1, self.rnnSize, 1], trainable=True)
            sru_h = tf.nn.depthwise_conv2d(sru_h, conv_filter, [1,1,1,1], padding='SAME',  name = 'QRNN_conv{}'.format(i))
            sru_h = tf.squeeze(sru_h, axis = [-2])
            sru_ = SRU_layer(self.rnnSize, batch_size=self.batch_size, fwidth=self.k_width, pool_type = 'ifo', zoneout=self.zoneout, name='QRNN_layer{}'.format(i), infer = tf.logical_not(self.is_train),skip = True, skip_embedding = False)
            print(sru_h)
            sru_h, last_state = sru_(sru_h)
            sru_h = tf.nn.dropout(sru_h, dropout_p, noise_shape = [tf.shape(sru_h)[0],1,tf.shape(sru_h)[2]])

        h_shape = tf.shape(sru_h)
        output_h = tf.reshape(sru_h, [-1, self.rnnSize])
        print(output_h)

        with tf.variable_scope('dense'): 
            dense = tf.layers.dense(output_h, self.outputSize, kernel_initializer=tf.random_uniform_initializer(-0.1,0.1))
            #dense = tf.layers.dense(output_h, self.outputSize)
        self.logit = tf.reshape(dense, [h_shape[0], h_shape[1], self.outputSize])
        self.logsoftmax = tf.nn.log_softmax(self.logit)
        self.loss = tf.nn.ctc_loss(
                inputs = self.logit,
                labels = self.label,
                sequence_length= self.seq_len,
                time_major = False
            )
        self.loss =tf.reduce_mean(self.loss)
        train_loss = self.loss 
        opt = tf.train.AdamOptimizer(self.lr) 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grad, var = zip(*opt.compute_gradients(train_loss))        
            clipped_gradients, _ = tf.clip_by_global_norm(grad,clip)
#            var_check = [tf.check_numerics(v, 'nan in var' + repr(v)) for v in var]
#            grad_check = [tf.check_numerics(g, 'nan in grad' + repr(g)) for g in clipped_gradients]
#            with tf.control_dependencies(var_check):
#                with tf.control_dependencies(grad_check):
            self.optimizer = opt.apply_gradients(zip(clipped_gradients, var))


        self.sentence, _ = tf.nn.ctc_greedy_decoder(tf.transpose(self.logit, (1,0,2)), self.seq_len)
        self.cer = tf.reduce_mean(tf.edit_distance(tf.cast(self.sentence[0], tf.int32),self.label))
        # last states to placeholder

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

    def test(self, dataset,  path='./model/hlstm'):
        loss = self.loss
        logit = self.logit
        logprob = self.logsoftmax
        cer = self.cer 
        test_loss = 0
        test_cer = 0
        test_cer2 = 0
        total_c = 0 
        config = tf.ConfigProto(device_count = {'GPU':0})
        sess = tf.Session(config=config)
        #sess = tf.Session()
        self.saver.restore(sess, save_path = path)
        p_dict = dict()
        for pp in tf.global_variables():
            print(pp.name)
            p_dict[pp.name] = sess.run(pp)
        #print(p_dict['SRU_conv0_filter:0'])
#        for i in range(6):
#            p_dict['SRU_conv{}_filter:0'.format(i)].tofile('SRU_conv{}_filter.csv'.format(i), sep=',', format='%10.5f' )
#        exit()
        #with open('./quant_weights_sru_char_all/float_params.pkl', 'wb') as f:
        #    pickle.dump(p_dict, f)
        #exit()
        st_time = time.time()
        dataset.set_mode('test')
        idx = 0
        logit_list = []
        len_list = []
        logprob_list = []
        trans_file = open(path + '_test_trans.txt','w+')
        result_file = open(path + '_test_result.txt','w+')

        charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ.' \n"
        while dataset.iter_flag():

            if idx % 100 == 0 and idx != 0:
                end_time = time.time()
                print('...... {:6d} / {:6d}, Time: {:.2f}, Loss: {:.4f}'.format(idx, dataset.n_batch, end_time-st_time, test_loss/idx))
                st_time = time.time()
            idx += 1
            batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape,  trans = dataset.get_data()
            ops = [loss, logit, cer, logprob]
            feed_dict = {
                    self.x: batch_x,
                    self.label: (sparse_indices, sparse_values, sparse_shape),
                    self.lr: 1.0,
                    self.seq_len: batch_seq_len,
                    self.is_train: False
                    }
            _loss, _logit, _cer, _logprob  = sess.run(ops, feed_dict)
            logit_list.append(_logit)
            logprob_list.append(_logprob)
            len_list.append(_logit.shape[1])
            test_loss += _loss
            #test_cer += _cer
            test_cer2 += _cer*sparse_shape[1]
            total_c += sparse_shape[1]
            _logit = np.argmax(_logit, axis = -1)
            print(self.greedy_decoding(_logit, charset))
            trans_file.write(trans[0])
            result_file.write(self.greedy_decoding(_logit, charset) + '\n')
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
                

    def train(self, dataset, lr, max_patience, max_change, batch_size, charset, regularizer=0.0, max_epoch=200, path='./model/hlstm'):
        optim = self.optimizer 
        loss = self.loss
        cer = self.cer
        logit = self.logit
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
        #print('model restored')
        #self.saver.restore(sess, './model/sru_conv_bias_ifo_am_k_1_layer_6_700_mid_conv_15_si284-154')
        
        cur_lr = lr
        for epoch in range(max_epoch):
            
            cur_slope = 1.0 + np.clip(epoch*0.2,0,4)
            if epoch == 0:
                cur_regularizer = 0.01
            else:
                cur_regularizer = 0
            
            print('... Epoch', epoch, status)
            print('... cur_slope:', cur_slope)

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
#            if epoch > 10:
            dataset.set_mode('train_under_1600')
#            else:
#                dataset.set_mode('train_under_800')
            if epoch != 0:
                while dataset.iter_flag():
                    if idx % 100 == 0 and idx != 0:
                        end_time = time.time()
                        print('...... {:6d} / {:6d}, Time: {:.2f}, Loss: {:.4f}'.format(idx, dataset.n_batch, end_time-st_time, train_loss/idx))
                        st_time = time.time()
                    idx +=1
                    batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape,  _ = dataset.get_data()
                    ops = [optim, loss, logit, cer]
                    feed_dict = {
                            self.x: batch_x,
                            self.label: (sparse_indices, sparse_values, sparse_shape),
                            self.lr: cur_lr,
                            self.seq_len: batch_seq_len,
                            self.is_train: True
                            }
                    _, _loss, _logit, _cer = sess.run(ops, feed_dict)

                    train_loss += _loss
                    train_cer += _cer
                    if np.isnan(_loss):
                        break

                if np.isnan(train_loss):
                    status = 'roll_back'
                    epoch -= 1
                    print('nan loss detected, roll back epoch')
                    continue
                train_loss /= dataset.n_batch
                train_cer /= dataset.n_batch
                train_curve.append(train_loss)
                g_norm_curve.append(train_g_norm)
                print('...... Train loss', train_loss)
                print('...... Train CER', train_cer)


            dataset.set_mode('valid')
            while dataset.iter_flag():
                if idx % 100 == 0 and idx != 0:
                    end_time = time.time()
                    print('...... {:6d} / {:6d}, Time: {:.2f}, Loss: {:.4f}'.format(idx, dataset.n_batch, end_time-st_time, valid_loss/idx))
                    st_time = time.time()

                idx += 1

                batch_x, batch_seq_len, sparse_indices, sparse_values, sparse_shape,  _ = dataset.get_data()
                boundary = np.zeros((batch_x.shape[0], batch_x.shape[1]),dtype=np.int32)
                ops = [loss, logit, cer]
                feed_dict = {
                        self.x: batch_x,
                        self.label: (sparse_indices, sparse_values, sparse_shape),
                        self.lr: cur_lr,
                        self.seq_len: batch_seq_len,
                        self.is_train: False
                        }
                _loss, _logit, _cer  = sess.run(ops, feed_dict)
                valid_loss += _loss
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
            print('...... Valid BPC, Valid CER, best CER', valid_loss, valid_cer, best_valid_loss)
            _logit = np.argmax(_logit, axis = -1)
            print(self.greedy_decoding(_logit, charset))

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
            np.save(path + '_gate_curve',np.array(g_norm_curve))
            np.save(path + '_cer_curve',np.array(cer_curve))



        return train_curve, valid_curve, cer_curve    





