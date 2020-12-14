import gc
from tqdm import tqdm
from scipy.stats import spearmanr
import numpy as np

import torch


def train_model(model, train_loader, optimizer, criterion, device='cuda'):

    model.train()
    avg_loss = 0.
    tk0 = tqdm(enumerate(train_loader))

    for idx, batch in tk0:

        input_ids, input_masks, input_segments, labels, _ = batch
        input_ids, input_masks, input_segments = input_ids.to(device), input_masks.to(device), input_segments.to(device)
        labels = labels.to(device)

        output_train = model(input_ids=input_ids.long(),
                             labels=None,
                             attention_mask=input_masks,
                             token_type_ids=input_segments,
                            )
        logits = output_train[0] #output preds

        loss = criterion(logits, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item() / len(train_loader)
        del input_ids, input_masks, input_segments, labels

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss

def val_model(model, criterion, val_loader, val_shape, num_labels, batch_size=8, device='cuda'):

    avg_val_loss = 0.
    model.eval() # eval mode

    valid_preds = np.zeros((val_shape, num_labels))
    original = np.zeros((val_shape, num_labels))

    tk0 = tqdm(enumerate(val_loader))
    with torch.no_grad():

        for idx, batch in tk0:
            input_ids, input_masks, input_segments, labels, _ = batch
            input_ids, input_masks, input_segments = input_ids.to(device), input_masks.to(device), input_segments.to(device)
            labels = labels.to(device)

            output_val = model(input_ids=input_ids.long(),
                             labels=None,
                             attention_mask=input_masks,
                             token_type_ids=input_segments,
                            )
            logits = output_val[0] #output preds

            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
            original[idx*batch_size : (idx+1)*batch_size]    = labels.detach().cpu().squeeze().numpy()

        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()

        # np.save("preds.npy", preds)
        # np.save("actuals.npy", original)

        rho_val = np.mean([spearmanr(original[:, i], preds[:,i]).correlation for i in range(preds.shape[1])])
        print('\r val_spearman-rho: %s' % (str(round(rho_val, 5))), end = 100*' '+'\n')

        for i in range(num_labels):
            # print(i, spearmanr(original[:,i], preds[:,i]))
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)
    return avg_val_loss, score/num_labels

# def compute_spearmanr(trues, preds):
#     rhos = []
#     for col_trues, col_pred in zip(trues.T, preds.T):
#         rhos.append(spearmanr(col_trues, col_pred + np.random.normal(0, 1e-7, col_pred.shape[0])).correlation)
#     return np.mean(rhos)

def predict_result(model, test_loader, num_labels, batch_size=32, device='cuda'):

    test_preds = np.zeros((len(test_loader), num_labels))

    model.eval();
    tk0 = tqdm(enumerate(test_loader))
    for idx, x_batch in tk0:
        with torch.no_grad():
            outputs = model(input_ids=x_batch[0].to(device),
                            labels=None,
                            attention_mask=x_batch[1].to(device),
                            token_type_ids=x_batch[2].to(device),
                           )
            predictions = outputs[0]
            test_preds[idx*batch_size : (idx+1)*batch_size] = predictions.detach().cpu().squeeze().numpy()

    output = torch.sigmoid(torch.tensor(test_preds)).numpy()
    return output

