import os
import time
from tqdm import tqdm

import torch

from src.evaluate import evaluate_lstm_crf, evaluate_bert, evaluate_bert_with_crf, compute_bert_loss


def train_lstm_with_crf(config, model, train_loader, dev_loader, optimizer, scheduler, logger):
    start_time = time.time()
    os.makedirs(config.weight_save_dir, exist_ok=True)
    model.to(config.device)

    global_step = 0
    total_loss = 0.
    logging_loss = 0.
    best_f1_score = 0.

    for epoch in range(config.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
            batch_inputs, batch_masks, batch_len, batch_labels = batch
            batch_inputs = batch_inputs.to(config.device)
            batch_masks = batch_masks.to(config.device)
            batch_labels = batch_labels.to(config.device)

            global_step += 1
            _, loss = model.forward_with_crf(batch_inputs, batch_masks, batch_len, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if global_step % config.period == 0:
                loss_scalar = (total_loss - logging_loss) / config.period
                logging_loss = total_loss
                logger.info(f"epoch: {epoch}, iter: {global_step}, loss: {loss_scalar:.4f}")

        precision, recall, f1 = evaluate_lstm_crf(config, model, dev_loader)
        logger.info(f"epoch {epoch}, evaluate result:\nprecision: {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}")

        if f1 > best_f1_score:
            best_f1_score = f1
            logger.info(f'Saving best model')
            path = os.path.join(config.weight_save_dir, 'BiLSTM_CRF.pt')
            torch.save(model.state_dict(), path)

    logger.info(f"training BiLSTM_CRF model takes total {int((time.time() - start_time) / 60)} m")


def train_bert(config, model, train_loader, dev_loader, optimizer, scheduler, logger):
    start_time = time.time()
    os.makedirs(config.weight_save_dir, exist_ok=True)
    model.to(config.device)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    global_step = 0
    total_loss = 0.
    logging_loss = 0.
    best_f1_score = 0.

    for epoch in range(config.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
            batch_inputs, batch_masks, batch_labels = batch
            batch_inputs = batch_inputs.to(config.device)
            batch_masks = batch_masks.to(config.device)
            batch_labels = batch_labels.to(config.device)

            global_step += 1
            logits = model(batch_inputs, batch_masks)
            loss = compute_bert_loss(logits, batch_labels, batch_masks, loss_fn)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if global_step % config.period == 0:
                loss_scalar = (total_loss - logging_loss) / config.period
                logging_loss = total_loss
                logger.info(f"epoch: {epoch}, iter: {global_step}, loss: {loss_scalar:.4f}")

        precision, recall, f1 = evaluate_bert(config, model, dev_loader)
        logger.info(f"epoch {epoch}, evaluate result:\nprecision: {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}")

        if f1 > best_f1_score:
            best_f1_score = f1
            logger.info(f'Saving best model')
            path = os.path.join(config.weight_save_dir, 'BERT.pt')
            torch.save(model.state_dict(), path)

    logger.info(f"training Bert model takes total {int((time.time() - start_time) / 60)} m")


def train_bert_with_crf(config, model, train_loader, dev_loader, optimizer, scheduler, logger, model_name=None):
    start_time = time.time()
    os.makedirs(config.weight_save_dir, exist_ok=True)
    model.to(config.device)

    global_step = 0
    total_loss = 0.
    logging_loss = 0.
    best_f1_score = 0.

    for epoch in range(config.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f'epoch {epoch}'):
            batch_inputs, batch_masks, batch_labels = batch
            batch_inputs = batch_inputs.to(config.device)
            batch_masks = batch_masks.to(config.device)
            batch_labels = batch_labels.to(config.device)

            global_step += 1
            loss = model(batch_inputs, batch_masks, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if global_step % config.period == 0:
                loss_scalar = (total_loss - logging_loss) / config.period
                logging_loss = total_loss
                logger.info(f"epoch: {epoch}, iter: {global_step}, loss: {loss_scalar:.4f}")

        precision, recall, f1 = evaluate_bert_with_crf(config, model, dev_loader)
        logger.info(f"epoch {epoch}, evaluate result:\nprecision: {precision:.2f}, recall {recall:.2f}, f1 {f1:.2f}")

        if f1 > best_f1_score:
            best_f1_score = f1
            logger.info(f'Saving best model')
            path = os.path.join(config.weight_save_dir, f'{model_name}.pt')
            torch.save(model.state_dict(), path)

    logger.info(f"training {model_name} model takes total {int((time.time() - start_time) / 60)} m")
