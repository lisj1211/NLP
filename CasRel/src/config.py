import json


class Config:
    def __init__(self, args):
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len
        self.bert_path = args.bert_path
        self.bert_dim = args.bert_dim
        self.max_norm = args.max_norm
        self.weight_decay = args.weight_decay
        self.lr_warmup = args.lr_warmup

        self.train_path = args.train_path
        self.test_path = args.test_path
        self.dev_path = args.dev_path
        self.rel_path = args.rel_path
        self.num_relations = len(json.load(open(self.rel_path, 'r', encoding='utf-8')))

        self.save_weights_dir = './weights/'
        self.weights_save_name = 'model.pt'

        self.period = 200
