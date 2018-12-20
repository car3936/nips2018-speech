class Config(object):

    def __init__(self):

        self.mode = 'tied'  # 'base', 'tied'
        if self.mode == 'tied':
            self.tied = True
        elif self.mode == 'base':
            self.tied = False
        else:
            raise ValueError('Not acceptable mode')

        self.torch_seed = 6420
        self.rnnSize = 840
        self.wiki_path = '/home/jhpark/data/ptb/'
        self.save_path = './best_model_lstm_{}'.format(self.rnnSize) + '_test_cwpg.pt'
        self.batch_size = 32
        self.sequence_length = 128
        self.eval_batch_size = 256

        self.max_epoch = 1000
        self.max_change = 6
        self.max_patience = 8

        self.initial_lr = 3e-4
        self.momentum = 0.9
        self.clip_norm = 4
        self.dropout = 0.75


