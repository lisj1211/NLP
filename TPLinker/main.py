from transformers import logging

from src.args import arg_parser
from src.models import TPLinkerBert
from src.log import TPLinkerLoging
from src.framework import TPLinkerFramework
from src.utils import set_seed

logging.set_verbosity_error()


def main():
    args = arg_parser()
    set_seed(1211)
    model = TPLinkerBert.from_pretrained(args.pretrained_bert_path, num_relations=args.num_rels)
    logger = TPLinkerLoging('TPLinker Model').logger

    tplinker_fw = TPLinkerFramework(model, args, logger)
    tplinker_fw.train()
    tplinker_fw.test()


if __name__ == '__main__':
    main()
