from tqdm import tqdm

import torch
from seqeval.metrics import precision_score, recall_score, f1_score


def evaluate_lstm_crf(config, model, dev_loader):
    y_true = []
    y_pred = []
    model.eval()
    for batch in tqdm(dev_loader, desc='evaluate'):
        with torch.no_grad():
            batch_inputs, batch_masks, batch_len, batch_labels = batch
            batch_inputs = batch_inputs.to(config.device)
            batch_masks = batch_masks.to(config.device)

            logits = model.forward(batch_inputs, batch_len)
            pred = model.crf.decode(logits, mask=batch_masks)

            tmp_true, tmp_pred = _lstm_output_convert(config.idx2label, batch_labels, batch_len, pred)
            y_true.extend(tmp_true)
            y_pred.extend(tmp_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def evaluate_bert(config, model, dev_loader):
    y_true = []
    y_pred = []
    model.eval()
    for batch in tqdm(dev_loader, desc='evaluate'):
        with torch.no_grad():
            batch_inputs, batch_masks, batch_labels = batch
            batch_inputs = batch_inputs.to(config.device)
            batch_masks = batch_masks.to(config.device)
            batch_labels = batch_labels.to(config.device)

            logits = model(batch_inputs, batch_masks)

            tmp_true, tmp_pred = _bert_output_convert(config.idx2label, batch_labels, batch_masks, logits)
            y_true.extend(tmp_true)
            y_pred.extend(tmp_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def evaluate_bert_with_crf(config, model, dev_loader):
    y_true = []
    y_pred = []
    model.eval()
    for batch in tqdm(dev_loader, desc='evaluate'):
        with torch.no_grad():
            batch_inputs, batch_masks, batch_labels = batch
            batch_inputs = batch_inputs.to(config.device)
            batch_masks = batch_masks.to(config.device)
            batch_labels = batch_labels.to(config.device)

            logits = model(batch_inputs, batch_masks)
            pred = model.crf.decode(logits, mask=batch_masks)

            tmp_true, tmp_pred = _bert_crf_output_convert(config.idx2label, batch_labels, batch_masks, pred)
            y_true.extend(tmp_true)
            y_pred.extend(tmp_pred)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1


def _lstm_output_convert(idx2label, batch_labels, batch_len, batch_pred):
    tmp_true = []
    tmp_pred = []
    for true_label, length, pred_label in zip(batch_labels, batch_len, batch_pred):
        true_label = true_label[:length]
        tmp_true.append([idx2label[t] for t in true_label.tolist()])
        tmp_pred.append([idx2label[p] for p in pred_label])

    return tmp_true, tmp_pred


def _bert_output_convert(idx2label, batch_labels, batch_masks, batch_logits):
    tmp_true = []
    tmp_pred = []
    batch_pred = torch.argmax(batch_logits, dim=-1)
    for true_label, masks, pred_label in zip(batch_labels, batch_masks, batch_pred):
        true_label = true_label[masks.nonzero()].squeeze()
        pred_label = pred_label[masks.nonzero()].squeeze()

        tmp_true.append([idx2label[t] for t in true_label.tolist()[1:-1]])
        tmp_pred.append([idx2label[p] for p in pred_label.tolist()[1:-1]])

    return tmp_true, tmp_pred


def _bert_crf_output_convert(idx2label, batch_labels, batch_masks, batch_pred):
    tmp_true = []
    tmp_pred = []
    for true_label, masks, pred_label in zip(batch_labels, batch_masks, batch_pred):
        true_label = true_label[masks.nonzero()].squeeze()

        tmp_true.append([idx2label[t] for t in true_label.tolist()[1:-1]])
        tmp_pred.append([idx2label[p] for p in pred_label[1:-1]])

    return tmp_true, tmp_pred


def compute_bert_loss(logits, labels, masks, loss_fn):
    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    masks = masks.view(-1)
    loss = loss_fn(logits, labels)
    loss = (loss * masks).sum() / masks.sum()

    return loss
