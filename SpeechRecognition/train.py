import argparse
import functools
import warnings

from src.trainer import SRTrainer
from src.utils.utils import add_arguments, print_arguments, set_seed
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',           str,    'configs/deepspeech2.yml',   '配置文件')
add_arg("use_gpu",           bool,   True,                        '是否使用GPU训练')
add_arg('augment_conf_path', str,    'configs/augmentation.json', '数据增强的配置文件，为json格式')
add_arg('save_model_path',   str,    'models/',                   '模型保存的路径')
add_arg('resume_model',      str,    None,                        '恢复训练，当为None则不使用预训练模型')
add_arg('pretrained_model',  str,    None,                        '预训练模型的路径，当为None则不使用预训练模型')
args = parser.parse_args()
print_arguments(args=args)

set_seed(1211)
# 获取训练器
trainer = SRTrainer(configs=args.configs, use_gpu=args.use_gpu)

trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model,
              augment_conf_path=args.augment_conf_path)