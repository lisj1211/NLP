import argparse
from itertools import product


def bilstm_crf_argparser():
    arg_parser = argparse.ArgumentParser(description='BiLSTM-CRF Controller')

    # Input
    arg_parser.add_argument('--train_path', type=str, default='./data/train.json', help="Path to train dataset")
    arg_parser.add_argument('--test_path', type=str, default='./data/dev.json', help="Path to test dataset")
    arg_parser.add_argument('--annotation_dtype', type=str, default='BIO', help="Data annotation format")

    # Path
    arg_parser.add_argument('--processed_train_data_path', type=str, default='./data/train.pkl',
                            help="Processed train data path")
    arg_parser.add_argument('--processed_test_data_path', type=str, default='./data/test.pkl',
                            help="Processed test data path")
    arg_parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl', help="Path to save vocabulary")
    arg_parser.add_argument('--weight_save_dir', type=str, default='./weight', help="Path to save model weight")

    # Model / Training
    arg_parser.add_argument('--max_vocab', type=int, default=50000, help="Max vocabulary size")
    arg_parser.add_argument('--min_freq', type=int, default=1, help="Min word frequency")
    arg_parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--clip_grad', type=int, default=5, help="Max grad norm")
    arg_parser.add_argument('--embedding_size', type=int, default=300, help="Embedding size")
    arg_parser.add_argument('--hidden_size', type=int, default=256, help="LSTM hidden size")
    arg_parser.add_argument('--drop_out', type=float, default=0.1, help="Dropout rate")
    arg_parser.add_argument('--period', type=int, default=20, help="Print evaluate period")

    arg_parser = arg_parser.parse_args()
    _add_common_args(arg_parser)

    return arg_parser


def bert_argparser():
    arg_parser = argparse.ArgumentParser(description='Bert Controller')

    # Input
    arg_parser.add_argument('--train_path', type=str, default='./data/train.json', help="Path to train dataset")
    arg_parser.add_argument('--test_path', type=str, default='./data/dev.json', help="Path to test dataset")
    arg_parser.add_argument('--annotation_dtype', type=str, default='BIO', help="Data annotation format")

    # Path
    arg_parser.add_argument('--bert_path', type=str, default='./src/bert-base-chinese', help="BertPreTrain Model Path")
    arg_parser.add_argument('--processed_train_data_path', type=str, default='./data/train.pkl',
                            help="Processed train data path")
    arg_parser.add_argument('--processed_test_data_path', type=str, default='./data/test.pkl',
                            help="Processed test data path")
    arg_parser.add_argument('--weight_save_dir', type=str, default='./weight', help="Path to save model weight")

    # Model / Training
    arg_parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    arg_parser.add_argument('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add_argument('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add_argument('--clip_grad', type=int, default=5, help="Max grad norm")
    arg_parser.add_argument('--drop_out', type=float, default=0.1, help="Dropout rate")
    arg_parser.add_argument('--period', type=int, default=20, help="Print evaluate period")

    arg_parser = arg_parser.parse_args()
    _add_common_args(arg_parser)

    return arg_parser


def _add_common_args(args):
    label2idx, idx2label = _get_label(args.annotation_dtype)
    args.label2idx = label2idx
    args.idx2label = idx2label
    args.num_class = len(idx2label)


def _get_label(dtype):
    labels = ['address', 'book', 'company', 'game', 'government',
              'movie', 'name', 'organization', 'position', 'scene']

    if dtype == 'BIOES':
        prefix = ['B-', 'I-', 'S-', 'E-']
    elif dtype == 'BIO':
        prefix = ['B-', 'I-']
    else:
        raise ValueError('Argument Error')

    label2idx = {"O": 0}
    for idx, (i, j) in enumerate(product(prefix, labels)):
        label2idx[i + j] = idx + 1
    idx2label = {_id: _label for _label, _id in list(label2idx.items())}

    return label2idx, idx2label
