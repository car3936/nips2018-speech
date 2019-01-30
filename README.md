# nips2018-speech


requirement: tensorflow >= 1.6

0. preprocessing (./wsj_data)

we converted .wv1 files in WSJ corpus to .wav files using sph2pipe. 

Features are extracted and saved as numpy file with preprocessing.py 


1. train (character-level model) 
 
python train_wsj_sru_ifo.py // train i-SRU

python train_wsj_sru_conv_ifo.py // train i-SRU + conv 

python train_wsj_lstm.py // train LSTM 


2. test (character-level model)

python test_wsj_sru_ifo.py // test i-SRU

python test_wsj_sru_conv_ifo.py // test i-SRU + conv

python test_wsj_lstm.py // test LSTM

3. train (wordpiece model)

python train_wsj_sru_wordpiece.py // train i-SRU + conv model

python train_wsj_lstm_wordpiece.py // train LSTM model

4. test (wordpiece model)

python test_wsj_wordpiece_m500_sru.py // test i-SRU + conv model

python test_wsj_wordpiece_m500_lstm.py // test LSTM model
