import json
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

from src.utils import move_dict_value_to_device, collate_fn
from src.dataset import MyDataset
from src.metrics import MetricsCalculator


class GlobalPointerFramework:
    def __init__(self, model, args, logger):
        self.model = model
        self.args = args
        self.logger = logger

    def save_model(self):
        torch.save(self.model.state_dict(), self.args.model_save_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.args.model_save_path, map_location=self.args.device))

    def configure_optimizer_and_scheduler(self, t_total):
        bert_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append((name, param))
            else:
                other_params.append((name, param))

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            # bert params
            {'params': [p for n, p in bert_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lr},
            {'params': [p for n, p in bert_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.lr},

            # other params
            {'params': [p for n, p in other_params if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay, 'lr': self.args.lr * 5},
            {'params': [p for n, p in other_params if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0, 'lr': self.args.lr * 5},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.args.lr, correct_bias=False, no_deprecation_warning=True
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

        return optimizer, scheduler

    @staticmethod
    def _multilabel_categorical_crossentropy(y_pred, y_true):
        """
        https://kexue.fm/archives/7359
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)

        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def train(self):
        time_start = time.time()
        self.logger.info(f'start train GlobalPointer model')

        self.logger.info(f'build dataset')
        all_data = json.load(open(self.args.train_path, 'r', encoding='utf-8'))
        train_data, dev_data = train_test_split(all_data, test_size=0.2, random_state=1211)

        train_dataset = MyDataset(data=train_data,
                                  pretrained_bert_path=self.args.pretrained_bert_path,
                                  ent2idx_path=self.args.ent2idx_path,
                                  max_sequence_length=self.args.max_sequence_length)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.train_batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=2,
                                  shuffle=True,
                                  drop_last=True)

        dev_dataset = MyDataset(data=dev_data,
                                pretrained_bert_path=self.args.pretrained_bert_path,
                                ent2idx_path=self.args.ent2idx_path,
                                max_sequence_length=self.args.max_sequence_length)
        dev_loader = DataLoader(dataset=dev_dataset,
                                batch_size=self.args.eval_batch_size,
                                collate_fn=collate_fn,
                                shuffle=False,
                                num_workers=2)
        metrics = MetricsCalculator()

        t_total = len(train_loader) * self.args.num_epochs
        optimizer, scheduler = self.configure_optimizer_and_scheduler(t_total)

        global_step = 0
        total_loss = 0.
        logg_loss = 0.
        best_f1 = 0.
        self.model.to(self.args.device)
        for epoch in range(self.args.num_epochs):
            self.model.train()
            for batch_data in tqdm(train_loader, desc=f'epoch {epoch}'):
                move_dict_value_to_device(batch_data, device=self.args.device)
                input_ids, attn_mask = batch_data['input_ids'], batch_data['attention_mask']
                logits = self.model(input_ids, attn_mask)

                loss = self._multilabel_categorical_crossentropy(logits, batch_data['h2t'])
                if loss < 100:
                    total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                global_step += 1
                if global_step % self.args.period == 0:
                    loss_scalar = (total_loss - logg_loss) / self.args.period
                    logg_loss = total_loss
                    self.logger.info(f'epoch: {epoch}, iter: {global_step}, loss: {loss_scalar:.4f}')

            _, _, val_f1 = self.evaluate(dev_loader, metrics)
            if val_f1 > best_f1:
                self.save_model()
                self.logger.info(f'save best model on epoch {epoch}')
                best_f1 = val_f1
            metrics.reset()

        time_end = time.time()
        self.logger.info(f'training GlobalPointer model takes total {int((time_end - time_start) / 60)} m')

    def evaluate(self, dev_loader, metrics):
        self.model.eval()
        with torch.no_grad():
            for batch_data in tqdm(dev_loader, desc='evaluate'):
                move_dict_value_to_device(batch_data, device=self.args.device)
                input_ids, attn_mask = batch_data['input_ids'], batch_data['attention_mask']
                logits = self.model(input_ids, attn_mask)

                correct_num, predict_num, gold_num = metrics.get_pred_count(logits, batch_data['h2t'])
                metrics.update(correct_num, predict_num, gold_num)

        precision, recall, f1 = metrics.get_entity_prf_scores()
        self.logger.info(f'eval entity result: precision {precision: .4f}, recall {recall: .4f}, f1 {f1: .4f}')
        return precision, recall, f1

    def test(self):
        self.logger.info(f'start test model')
        with open(self.args.test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        max_tok_num = 0
        for sample in tqdm(data, desc="Calculate the max token number"):
            max_tok_num = max(len(sample['tokens']), max_tok_num)

        test_data = json.load(open(self.args.test_path, 'r', encoding='utf-8'))
        test_dataset = MyDataset(data=test_data,
                                 pretrained_bert_path=self.args.pretrained_bert_path,
                                 ent2idx_path=self.args.ent2idx_path,
                                 max_sequence_length=self.args.max_sequence_length)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.args.eval_batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=2,
                                 shuffle=False)

        self.load_model()
        self.model.to(self.args.device)

        metrics = MetricsCalculator()
        self.evaluate(test_loader, metrics)

    def predict(self, text):
        ent2idx = json.load(open(self.args.ent2idx_path, 'r', encoding='utf-8'))
        idx2ent = {idx: ent for ent, idx in ent2idx.items()}
        tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_bert_path)
        tokens = tokenizer.tokenize(text)
        input_dic = tokenizer.encode_plus(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.args.max_sequence_length,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = input_dic['input_ids']
        attention_mask = input_dic['attention_mask']

        self.load_model()
        self.model.to(self.args.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(input_ids, attention_mask)
        pred = pred.cpu().numpy()
        pred_set = set()
        for batch, ent_idx, token_start, token_end in zip(*np.where(pred > 0)):
            ent_type = idx2ent[ent_idx]
            ent_token_ids = input_ids[0][token_start:token_end]
            ent_text = ''.join(tokenizer.convert_ids_to_tokens(ent_token_ids))
            pred_set.add((ent_type, ent_text))

        print(pred_set)
