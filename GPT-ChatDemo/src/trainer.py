import json
import os
import shutil
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup, BertTokenizer

from src.logger import setup_logger
from src.utils import print_arguments
from src.dataset import MyDataset

logger = setup_logger(__name__)


class Trainer:
    def __init__(self, args, use_gpu=True):
        """
        :param args: 配置文件
        :param use_gpu: 是否使用GPU训练模型
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU不可用'
            self.device = torch.device('cuda')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device('cpu')
        print_arguments(args=args)
        self.args = args
        self.use_gpu = use_gpu
        self.model = None

    def __setup_dataloader(self, is_train=False):
        if is_train:
            self.train_dataset = MyDataset(data_path=self.args.data_path,
                                           vocab_path=self.args.vocab_path,
                                           max_seq_len=self.args.max_seq_len)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           batch_size=self.args.batch_size,
                                           collate_fn=self.train_dataset.collate_fn,
                                           prefetch_factor=self.args.prefetch_factor,
                                           num_workers=self.args.num_workers,
                                           drop_last=True,
                                           shuffle=True)

    def __setup_model(self, is_train=False):
        model_config = GPT2Config.from_json_file(self.args.model_config)
        self.model = GPT2LMHeadModel(config=model_config)
        self.model.to(self.device)
        if is_train:
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.args.weight_decay, 'lr': self.args.lr},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0, 'lr': self.args.lr},
            ]

            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, correct_bias=False,
                                   no_deprecation_warning=True)
            t_total = len(self.train_loader) // self.args.accum_grad * self.args.max_epoch
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
            )

    def __load_pretrained(self, pretrained_model):
        # 加载预训练模型
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pt')
            assert os.path.exists(pretrained_model), f"{pretrained_model} 模型不存在！"
            model_dict = self.model.state_dict()
            model_state_dict = torch.load(pretrained_model)
            # 特征层
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            self.model.load_state_dict(model_state_dict, strict=False)
            logger.info(f'成功加载预训练模型：{pretrained_model}')

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        save_model_name = 'GPT-chat'
        last_model_dir = os.path.join(save_model_path, save_model_name, 'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pt'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pt'))):
            # 判断从指定resume_model恢复训练，还是last_model恢复训练
            if resume_model is None:
                resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pt')), "模型参数文件不存在！"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pt')), "优化方法参数文件不存在！"
            state_dict = torch.load(os.path.join(resume_model, 'model.pt'))
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
            logger.info(f'成功恢复模型参数和优化方法参数：{resume_model}')
        return last_epoch

    # 保存模型
    def __save_checkpoint(self, save_model_path, epoch_id, best_model=False):
        save_model_name = 'GPT-chat'
        if best_model:
            model_path = os.path.join(save_model_path, save_model_name, 'best_model')
        else:
            model_path = os.path.join(save_model_path, save_model_name, 'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
        torch.save(self.model.state_dict(), os.path.join(model_path, 'model.pt'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            state = {"last_epoch": epoch_id}
            json.dump(state, f)
        if not best_model:
            last_model_path = os.path.join(save_model_path, save_model_name, 'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # 删除旧的模型
            old_model_path = os.path.join(save_model_path, save_model_name, 'epoch_{}'.format(epoch_id - 1))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('已保存模型：{}'.format(model_path))

    def __train_epoch(self, epoch_id):
        accum_grad = self.args.accum_grad
        grad_clip = self.args.grad_clip
        train_times, batch_times = [], []
        self.model.train()
        for batch_id, batch in enumerate(tqdm(self.train_loader, desc=f'epoch:{epoch_id}')):
            inputs, labels = batch
            start_step = time.time()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss / accum_grad
            loss.backward()
            # 执行一次梯度计算
            if batch_id % accum_grad == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                if torch.isfinite(grad_norm):
                    self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.train_step += 1
            batch_times.append((time.time() - start_step) * 1000)

            if batch_id % self.args.log_interval == 0:
                logger.info(f'loss: {loss.cpu().detach().numpy():.5f}, '
                            f'learning_rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'batch_cost: {(sum(batch_times) / len(batch_times) / 1000):.4f}, ')

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None):
        """
        训练模型
        :param save_model_path: 模型保存的路径
        :param resume_model: 恢复训练，当为None则不使用预训练模型
        :param pretrained_model: 预训练模型的路径，当为None则不使用预训练模型
        """
        start_time = time.time()
        # 获取数据
        self.__setup_dataloader(is_train=True)
        logger.info(f'训练数据大小：{len(self.train_dataset)}')
        # 获取模型
        self.__setup_model(is_train=True)
        self.__load_pretrained(pretrained_model=pretrained_model)
        self.print_num_parameters()
        # 加载恢复模型
        last_epoch = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        self.train_step = 0
        last_epoch += 1
        # 开始训练
        for epoch_id in range(last_epoch, self.args.max_epoch):
            epoch_id += 1
            self.__train_epoch(epoch_id=epoch_id)
            logger.info('=' * 70)
            # 保存模型
            self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id)
        logger.info(f'training model cost {(time.time() - start_time) / 60:.1f}')

    def start_chat(self, save_model_path='models/'):
        """
        导出预测模型
        :param save_model_path: 模型待保存的路径
        :return:
        """
        # 获取模型
        self.__setup_model()
        # 加载预训练模型
        assert os.path.exists(save_model_path), f"{save_model_path} 模型不存在！"
        if torch.cuda.is_available() and self.use_gpu:
            model_state_dict = torch.load(save_model_path)
        else:
            model_state_dict = torch.load(save_model_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict)
        logger.info('成功恢复模型参数和优化方法参数：{}'.format(save_model_path))
        self.model.eval()
        tokenizer = BertTokenizer(vocab_file=self.args.vocab_path, padding_side='left',
                                  sep_token='[SEP]', pad_token='[PAD]', cls_token='[CLS]')
        sep_token_id = tokenizer.sep_token_id

        history = []
        print('开始和chatbot聊天，输入CTRL + C以退出')

        while True:
            try:
                text = input("user:")
                # text = "你好"
                text_ids = tokenizer.encode(text, add_special_tokens=False)
                history.append(text_ids)
                input_ids = [tokenizer.cls_token_id]  # 每个input以[CLS]为开头

                for history_id, history_utr in enumerate(history[-self.args.max_history_len:]):
                    input_ids.extend(history_utr)
                    input_ids.append(sep_token_id)
                input_len = len(input_ids)
                input_ids = torch.tensor(input_ids).long().to(self.device)
                input_ids = input_ids.unsqueeze(0)
                input_dic = {
                    'input_ids': input_ids,
                    'do_sample': True,
                    'max_new_tokens': self.args.max_len,
                    'temperature': self.args.temperature,
                    'top_k': self.args.top_k,
                    'top_p': self.args.top_p,
                    'repetition_penalty': self.args.repetition_penalty,
                    'eos_token_id': tokenizer.sep_token_id,
                    'pad_token_id': tokenizer.sep_token_id
                }
                output = self.model.generate(**input_dic)[0]
                response = output[input_len:-1] if output[-1] == sep_token_id else output[input_len:]
                history.append(response)
                text = tokenizer.convert_ids_to_tokens(response)
                print('chatbot: ' + ''.join(text))
            except KeyboardInterrupt:
                break

    def print_num_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f'total parameters: {total_params}, total_trainable_params: {total_trainable_params}')
