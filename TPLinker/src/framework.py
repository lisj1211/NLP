import json
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from src.utils import move_dict_value_to_device, collate_fn
from src.loss import MyLoss
from src.dataset import MyDataset
from src.metrics import MetricsCalculator


class TPLinkerFramework:
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

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, correct_bias=False, no_deprecation_warning=True)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(self.args.warmup_proportion * t_total), num_training_steps=t_total
        )

        return optimizer, scheduler

    @staticmethod
    def _loss_fn(y_pred, y_true, num_class):
        y_pred = y_pred.view(-1, y_pred.size()[-1])
        y_true = y_true.view(-1)
        weight = []
        for i in range(num_class):
            weight.append(((y_true == i).float().sum() / y_true.shape[0]).item())
        return F.cross_entropy(y_pred, y_true)

    def _get_loss_weight(self, current_step):
        z = (2 * self.args.num_rels + 1)
        total_steps = self.args.loss_weight_recover_steps + 1  # + 1 avoid division by zero error
        w_ent = max(1 / z + 1 - current_step / total_steps, 1 / z)
        w_rel = min((self.args.num_rels / z) * current_step / total_steps, (self.args.num_rels / z))
        return w_ent, w_rel

    def train(self, dev=True):
        time_start = time.time()
        self.logger.info(f'start train TPLinker model')

        self.logger.info(f'build dataset')
        train_dataset = MyDataset(data_path=self.args.train_path,
                                  pretrained_bert_path=self.args.pretrained_bert_path,
                                  rel_path=self.args.rel_path,
                                  max_sequence_length=self.args.train_max_length,
                                  window_size=self.args.train_window_size)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=self.args.train_batch_size,
                                  collate_fn=collate_fn,
                                  num_workers=2,
                                  shuffle=True,
                                  drop_last=True)
        dev_loader = None
        metrics = None
        if dev:
            dev_dataset = MyDataset(data_path=self.args.dev_path,
                                    pretrained_bert_path=self.args.pretrained_bert_path,
                                    rel_path=self.args.rel_path,
                                    max_sequence_length=self.args.train_max_length,
                                    window_size=self.args.train_window_size)
            dev_loader = DataLoader(dataset=dev_dataset,
                                    batch_size=self.args.eval_batch_size,
                                    collate_fn=collate_fn,
                                    shuffle=False,
                                    num_workers=2)
            metrics = MetricsCalculator(self.args.rel_path, self.args.train_max_length)

        t_total = len(train_loader) * self.args.num_epochs
        optimizer, scheduler = self.configure_optimizer_and_scheduler(t_total)
        # loss_fn = MyLoss()  # focal_loss

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
                h2t_pred, h2h_pred, t2t_pred = self.model(input_ids, attn_mask)

                h2t_loss = self._loss_fn(h2t_pred, batch_data['h2t'], num_class=2)
                h2h_loss = self._loss_fn(h2h_pred, batch_data['h2h'], num_class=3)
                t2t_loss = self._loss_fn(t2t_pred, batch_data['t2t'], num_class=3)
                w_ent, w_rel = self._get_loss_weight(global_step)
                loss = w_ent * h2t_loss + w_rel * h2h_loss + w_rel * t2t_loss
                # loss = loss_fn(h2t_pred, h2h_pred, t2t_pred, batch_data['h2t'], batch_data['h2h'], batch_data['t2t'])
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

            if dev:
                _, _, val_f1 = self.evaluate(dev_loader, metrics)
                if val_f1 > best_f1:
                    self.save_model()
                    self.logger.info(f'save best model on epoch {epoch}')
                    best_f1 = val_f1
                metrics.reset()

        time_end = time.time()
        self.logger.info(f'training TPLinker model takes total {int((time_end - time_start) / 60)} m')

    def evaluate(self, dev_loader, metrics):
        self.model.eval()
        with torch.no_grad():
            for batch_data in tqdm(dev_loader, desc='evaluate'):
                move_dict_value_to_device(batch_data, device=self.args.device)
                input_ids, attn_mask = batch_data['input_ids'], batch_data['attention_mask']
                h2t_pred, h2h_pred, t2t_pred = self.model(input_ids, attn_mask)

                head2tail_acc = metrics.get_sample_accuracy(h2t_pred, batch_data['h2t'])
                head2head_acc = metrics.get_sample_accuracy(h2h_pred, batch_data['h2h'])
                tail2tail_acc = metrics.get_sample_accuracy(t2t_pred, batch_data['t2t'])

                samples = [json.loads(b) for b in batch_data['sample']]
                correct_num, predict_num, gold_num = metrics.get_rel_count(samples, h2t_pred, h2h_pred, t2t_pred)
                metrics.update(head2tail_acc, head2head_acc, tail2tail_acc, correct_num, predict_num, gold_num)

        h2t_acc, h2h_acc, t2t_acc = metrics.get_entity_acc(len(dev_loader))
        precision, recall, f1 = metrics.get_relation_prf_scores()
        self.logger.info(f'eval entity result: h2t_acc {h2t_acc: .4f}, h2h_acc {h2h_acc: .4f}, t2t_acc {t2t_acc: .4f}')
        self.logger.info(f'eval relation result: precision {precision: .4f}, recall {recall: .4f}, f1 {f1: .4f}')
        return precision, recall, f1

    def test(self):
        self.logger.info(f'start test model')
        with open(self.args.test_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        max_tok_num = 0
        for sample in tqdm(data, desc="Calculate the max token number"):
            max_tok_num = max(len(sample['tokens']), max_tok_num)

        test_dataset = MyDataset(data_path=self.args.test_path,
                                 pretrained_bert_path=self.args.pretrained_bert_path,
                                 rel_path=self.args.rel_path,
                                 max_sequence_length=max_tok_num)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=self.args.eval_batch_size,
                                 collate_fn=collate_fn,
                                 num_workers=2,
                                 shuffle=False)

        self.load_model()
        self.model.to(self.args.device)

        metrics = MetricsCalculator(self.args.rel_path, max_tok_num)
        self.evaluate(test_loader, metrics)
