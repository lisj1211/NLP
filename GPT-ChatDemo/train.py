import argparse

from src.trainer import Trainer

parser = argparse.ArgumentParser(description=__doc__)
# Path
parser.add_argument('--data_path', type=str, default='data/train.txt', help='训练数据路径')
parser.add_argument('--vocab_path', type=str, default='configs/vocab.txt', help='Tokenizer词典文件')
parser.add_argument('--model_config', type=str, default='configs/config.json', help='GPT配置文件')
# Data Preprocess
parser.add_argument('--max_seq_len', type=int, default=200, help='GPT输入最大长度')
# Train
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--prefetch_factor', type=int, default=2, help='加速dataloader')
parser.add_argument('--num_workers', type=int, default=2, help='加速dataloader')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='权重衰减')
parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
parser.add_argument('--max_epoch', type=int, default=50, help='训练迭代次数')
parser.add_argument('--accum_grad', type=int, default=1, help='梯度累积')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size')
parser.add_argument('--warmup_proportion', type=float, default=0.1, help='学习率预热比例')
parser.add_argument('--grad_clip', type=float, default=2, help='梯度截断')
# Show
parser.add_argument('--log_interval', type=int, default=500, help='打印中间结果间隔')
args = parser.parse_args()


def main(args):
    trainer = Trainer(args, use_gpu=args.use_gpu)
    trainer.train()


if __name__ == '__main__':
    main(args)
