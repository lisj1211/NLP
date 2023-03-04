import argparse

from transformers import logging

from src.trainer import Trainer
logging.set_verbosity_error()

parser = argparse.ArgumentParser(description=__doc__)
# Model Config
parser.add_argument('--model_config', type=str, default='configs/config.json', help='GPT配置文件')
parser.add_argument('--vocab_path', type=str, default='configs/vocab.txt', help='Tokenizer词典文件')
parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
parser.add_argument('--model_path', type=str, default='models/GPT-chat/epoch_50/model.pt', help='保存的模型路径')
# Generate Config
parser.add_argument('--max_history_len', type=int, default=3, help='考虑dialogue history的最大长度')
parser.add_argument('--max_len', type=int, default=25, help='每个utterance的最大长度,超过指定长度则进行截断')
parser.add_argument('--repetition_penalty', default=1.0, type=float, help='生成的对话重复惩罚参数')
parser.add_argument('--temperature', default=1, type=float, help='控制概率的平滑程度')
parser.add_argument('--top_k', default=8, type=int, help='最高k选1')
parser.add_argument('--top_p', default=0.92, type=float, help='最高积累概率')
args = parser.parse_args()


def main(args):
    trainer = Trainer(args, use_gpu=args.use_gpu)
    trainer.start_chat(save_model_path=args.model_path)


if __name__ == '__main__':
    main(args)
