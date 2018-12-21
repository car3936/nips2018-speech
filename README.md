# nips2018-speech


requirement: tensorflow >= 1.6

0. preprocessing 

1. train (character-level model) 
 
python train_wsj_sru_ifo.py// train i-SRU
python train_wsj_sru_conv_ifo.py // train i-SRU + conv 
python train_wsj_lstm.py


// train LSTM

2. test (character-level model)
python test_wsj_sru_ifo.py// test i-SRU
python test_wsj_sru_conv_ifo.py // test i-SRU + conv
python test_wsj_lstm.py// test LSTM

3. train (wordpiece model)

// i-SRU + conv 
// LSTM    

4. test (wordpiece model)

// i-SRU + conv 
// LSTM
