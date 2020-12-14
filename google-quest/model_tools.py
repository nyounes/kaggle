import time
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr

import torch
from torch.optim import lr_scheduler

from logger import logger


def train_model(model, train_loader, valid_loader, n_epochs=5, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr)
    patience = 4

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='mean').cuda()
    best_score = 0
    best_epoch = 0
    best_loss = 0
    best_val_loss = 0

    for epoch in range(n_epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.

        for question, answer, title, use_emb_q, use_emb_a, use_emb_t, dist_feature, y_batch in tqdm(train_loader, disable=True):
            question = question.long().cuda()
            answer = answer.long().cuda()
            title = title.long().cuda()
            use_emb_q = use_emb_q.cuda()
            use_emb_a = use_emb_a.cuda()
            use_emb_t = use_emb_t.cuda()
            dist_feature = dist_feature.cuda()

            y_batch = y_batch.cuda()
            y_pred = model(question, answer, title, use_emb_q, use_emb_a, use_emb_t, dist_feature)

            loss = loss_fn(y_pred.double(), y_batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            avg_loss += loss.item() / len(train_loader)

        model.eval()

        avg_val_loss = 0.
        preds = []
        original = []
        for i, (question, answer, title, use_emb_q, use_emb_a, use_emb_t, dist_feature, y_batch) in enumerate(valid_loader):
            question = question.long().cuda()
            answer = answer.long().cuda()
            title = title.long().cuda()
            use_emb_q = use_emb_q.cuda()
            use_emb_a = use_emb_a.cuda()
            use_emb_t = use_emb_t.cuda()
            dist_feature = dist_feature.cuda()

            y_batch = y_batch.cuda()
            y_pred = model(question, answer, title, use_emb_q, use_emb_a, use_emb_t,
                           dist_feature).detach()

            avg_val_loss += loss_fn(y_pred.double(), y_batch).item() / len(valid_loader)
            preds.append(y_pred.cpu().numpy())
            original.append(y_batch.cpu().numpy())

        score = 0
        for i in range(30):
            score += np.nan_to_num(
                spearmanr(np.concatenate(original)[:, i], np.concatenate(preds)[:, i]).correlation / 30)
        elapsed_time = time.time() - start_time

        logger.info('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t spearman={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, avg_val_loss, score, elapsed_time))

        scheduler.step(avg_val_loss)

        valid_score = score
        if valid_score > best_score:
            best_score = valid_score
            best_epoch = epoch
            best_train_loss = avg_loss
            best_val_loss = avg_val_loss
            p = 0

        # check if validation loss didn't improve
        if valid_score <= best_score:
            p += 1
            # print(f'{p} epochs of non improving score')
            if p > patience:
                # print('Stopping training')
                stop = True
                break

    return model, best_score, best_epoch, best_train_loss, best_val_loss


def make_prediction(test_loader, model):
    prediction = np.zeros((len(test_loader.dataset), 30))
    model.eval()
    for i, (question, answer, title, use_emb_q, use_emb_a, use_emb_t, dist_feature, _) in enumerate(test_loader):

        start_index = i * test_loader.batch_size
        end_index = min(start_index + test_loader.batch_size, len(test_loader.dataset))
        question = question.long().cuda()
        answer = answer.long().cuda()
        title = title.long().cuda()
        use_emb_q = use_emb_q.cuda()
        use_emb_a = use_emb_a.cuda()
        use_emb_t = use_emb_t.cuda()
        dist_feature = dist_feature.cuda()
        y_pred = model(question, answer, title, use_emb_q, use_emb_a, use_emb_t, dist_feature).detach()
        y_pred = torch.sigmoid(y_pred)
        prediction[start_index:end_index, :] += y_pred.detach().cpu().numpy()

    return prediction
