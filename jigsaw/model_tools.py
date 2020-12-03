from logger import logger
import numpy as np
import torch
from fastai.callbacks import TrainingPhase, GeneralScheduler, SaveModelCallback, EarlyStoppingCallback


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def save_model_without_embedding(model, model_name):
    logger.info("saving model in {}".format(model_name))
    temp_dict = model.state_dict()
    del temp_dict['embedding.weight']
    torch.save(temp_dict, model_name)


def moving_average(net1, net2, alpha=1):
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


def predict(test, learner, batch_size, output_dim):
    learner.model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
        test_preds = np.zeros((len(test), output_dim))
        for i, x_batch in enumerate(test_loader):
            x = x_batch[0].cuda()
            x_features = x_batch[1].cuda()
            y_pred = sigmoid(learner.model(x, x_features).detach().cpu().numpy())
            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred
    return test_preds


def predict_with_model(test, model, batch_size, output_dim):
    model.eval()
    with torch.no_grad():
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
        test_preds = np.zeros((len(test), output_dim))
        for i, x_batch in enumerate(test_loader):
            x = x_batch[0]
            x_features = x_batch[1]
            y_pred = sigmoid(model(x, x_features).detach().cpu().numpy())
            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred
    return test_preds


def train_model(learn, lr=0.001, lr_decay=0.8, batch_size=512, n_epochs=20, model_name='fastai_'):
    n = len(learn.data.train_dl)
    phases = [(TrainingPhase(n).schedule_hp('lr', lr * (lr_decay ** (i)))) for i in range(n_epochs)]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)

    learn.fit(n_epochs,
              callbacks=[SaveModelCallback(learn, name=model_name),
                         EarlyStoppingCallback(learn, min_delta=0.001, patience=5)])


def train_model_per_epoch(learn, test, output_dim, model_idx, lr=0.001, lr_decay=0.6, batch_size=512, n_epochs=20,
                          early_stopping=5, save_models='all', model_name='fastai_'):
    all_test_preds = []
    best_loss = -1
    best_epoch = 1
    for epoch in range(n_epochs):

        learn.fit(1, lr=lr * (lr_decay ** epoch))

        current_val_loss = learn.recorder.val_losses[0]
        test_preds = predict(test, learn, batch_size=256, output_dim=output_dim)
        all_test_preds.append(test_preds)
        if save_models == 'all':
            save_model_without_embedding(learn.model, model_name + str(model_idx * n_epochs + epoch) + ".pt")
        if best_loss == -1 or current_val_loss < best_loss:
            best_epoch = epoch
            best_loss = current_val_loss
        else:
            if epoch - best_epoch == early_stopping:
                break

        logger.info("best epoch: {}, best loss: {:4.5f}".format(best_epoch, best_loss))

    if save_models == 'last':
        save_model_without_embedding(learn.model, model_name + str(model_idx * n_epochs + epoch) + ".pt")
    return all_test_preds
