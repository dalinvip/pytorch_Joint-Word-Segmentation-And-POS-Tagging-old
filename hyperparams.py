import torch
import random
torch.manual_seed(121)
random.seed(121)
# random seed
seed_num = 233


class Hyperparams():
    def __init__(self):

        # data path
        # self.train_path = "./pos_test_data/train.ctb60.pos.hwc"
        # self.dev_path = "./pos_test_data/dev.ctb60.pos.hwc"
        # self.test_path = "./pos_test_data/test.ctb60.pos.hwc"
        self.train_path = "./posdata/train.ctb60.pos.hwc"
        self.dev_path = "./posdata/dev.ctb60.pos.hwc"
        self.test_path = "./posdata/test.ctb60.pos.hwc"

        self.learning_rate = 0.001
        self.learning_rate_decay = 0.9   # value is 1 means not change lr
        # self.learning_rate_decay = 1   # value is 1 means not change lr
        self.epochs = 10
        self.batch_size = 16
        self.log_interval = 1
        self.test_interval = 100
        self.save_interval = 100
        self.save_dir = "snapshot"
        self.shuffle = True
        self.epochs_shuffle = True
        self.dropout = 0.6
        self.dropout_embed = 0.6
        self.max_norm = None
        self.clip_max_norm = 5
        self.static = False
        # model
        self.LSTM = False
        self.BiLSTM_1 = False
        # select optim algorhtim to train
        self.Adam = True
        self.SGD = False
        self.Adadelta = False
        self.optim_momentum_value = 0.9
        # min freq to include during built the vocab, default is 1
        self.min_freq = 1
        # word_Embedding
        self.word_Embedding = True
        self.word_Embedding_Path = "./word2vec/glove.sentiment.conj.pretrained.txt"
        self.fix_Embedding = False
        self.embed_dim = 300
        self.embed_char_dim = 200
        self.embed_bichar_dim = 200
        # word_Embedding_Path = "./word2vec/glove.840B.300d.handled.Twitter.txt"
        self.char_Embedding = True
        self.char_Embedding_path = "./word_embedding/char.vec"
        self.bichar_Embedding = True
        self.bichar_Embedding_Path = "./word_embedding/bichar.vec"

        self.rnn_hidden_dim = 200
        self.hidden_size = 200
        self.rnn_num_layers = 1
        self.gpu_device = 0
        self.use_cuda = False
        self.snapshot = None
        self.num_threads = 1
        # whether to init w
        self.init_weight = True
        self.init_weight_value = 6.0
        # L2 weight_decay
        # self.weight_decay = 1e-9   # default value is zero in Adam SGD
        self.weight_decay = 0   # default value is zero in Adam SGD
        # whether to delete the model after test acc so that to save space
        self.rm_model = True



