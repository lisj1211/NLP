# -*- coding:utf-8 -*-


class Config:
    def __init__(self, args):
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_grad_norm = args.max_norm
        self.weight_decay = args.weight_decay
        self.lr_warmup = args.lr_warmup
        self.bert_path = args.bert_path
        self.bert_dim = args.bert_dim
        self.size_embedding = args.size_embedding
        self.prop_drop = args.prop_drop
        self.freeze_bert = args.freeze_bert
        self.neg_entity_count = args.neg_entity_count
        self.neg_rel_count = args.neg_rel_count
        self.max_span_size = args.max_span_size

        self.train_path = args.train_path
        self.test_path = args.test_path
        self.dev_path = args.dev_path
        self.type_path = args.type_path

        self.save_weights_dir = './weights/'
        self.weights_save_name = 'model.pt'

        self.period = 200
