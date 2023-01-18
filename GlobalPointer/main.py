from transformers import logging

from src.args import arg_parser
from src.models import GlobalPointer
from src.log import GlobalPointerLoging
from src.framework import GlobalPointerFramework
from src.utils import set_seed

logging.set_verbosity_error()


def main():
    args = arg_parser()
    set_seed(1211)
    model = GlobalPointer.from_pretrained(
        args.pretrained_bert_path, ent_type_size=args.ent_type_size, inner_dim=args.inner_dim
    )
    logger = GlobalPointerLoging('GlobalPointer Model').logger

    GP_fw = GlobalPointerFramework(model, args, logger)
    GP_fw.train()
    GP_fw.test()
    GP_fw.predict("诺贝尔经济学奖得主迈伦·斯科尔斯日前在北京大学发表演讲时指出，")


if __name__ == '__main__':
    main()
