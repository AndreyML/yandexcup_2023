import copy
from functools import partial
import json

import torch
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import average_precision_score
from hyperopt import fmin, tpe, Trials
import wandb

from settings import RANDOM_STATE, CUDA_DEV


def train_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    scheduler,
    device=CUDA_DEV,
    scaler=None,
):
    model.to(device)
    model.train()
    total_loss = 0
    dl_size = len(data_loader)
    preds = []
    targets = []

    for batch in tqdm(data_loader):
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()
        logits = model(batch)
        probs = torch.sigmoid(logits)
        loss = loss_function(logits, batch["labels"])
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        preds.append(probs.detach().cpu().numpy())
        targets.append(batch["labels"].cpu().numpy())
    if scheduler is not None:
        scheduler.step()

    preds = np.vstack(preds)
    targets = np.vstack(targets)
    ap = sum(
        [
            average_precision_score(y_true, y_preds)
            for y_true, y_preds in zip(targets, preds)
        ]
    ) / len(preds)
    metrics = {"Train Loss": total_loss / dl_size, "Train AveragePrecision": ap}
    return metrics


def eval_epoch(model, data_loader, loss_function, device=CUDA_DEV):
    model.to(device)
    model.eval()
    total_loss = 0
    preds = []
    targets = []
    dl_size = len(data_loader)

    for batch in tqdm(data_loader):
        for key in batch:
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds.append(probs.detach().cpu().numpy())
            targets.append(batch["labels"].detach().cpu().numpy())

        loss = loss_function(logits, batch["labels"])
        total_loss += loss.item()

    preds = np.vstack(preds)
    targets = np.vstack(targets)
    ap = sum(
        [
            average_precision_score(y_true, y_preds)
            for y_true, y_preds in zip(targets, preds)
        ]
    ) / len(preds)
    metrics = {"Val Loss": total_loss / dl_size, "Val AveragePrecision": ap}
    return metrics


def single_model(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    device=torch.device("cpu"),
    epochs: int = 8,
    start_epoch=0,
):
    loss_function.to(device)
    model.to(device)
    #wandb.init(project="YandexMLCup")
    for epoch_i in range(0, epochs):
        if epoch_i >= start_epoch:
            train_metrics = train_epoch(
                model, train_loader, loss_function, optimizer, scheduler, device
            )
            '''train = {
                "train/train_loss": train_metrics["Train Loss"],
                "train/epoch": epoch_i,
                "train/ap": train_metrics["Train AveragePrecision"],
            }
            val = {
                "val/val_loss": eval_metrics["Val Loss"],
                "val/epoch": epoch_i,
                "val/ap": train_metrics["Val AveragePrecision"],
            }
            wandb.log({**train, **val})
            '''

            eval_metrics = eval_epoch(model, val_loader, loss_function, device)
            print("EPOCH", epoch_i)
            print(train_metrics)
            print(eval_metrics)
    #wandb.finish()


def train_model_early_stopping(
    model,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    device=torch.device("cpu"),
    early_stopping: int = 2,
    eps: int = 1e-3,
):
    loss_function.to(device)
    model.to(device)
    es = early_stopping
    max_ap = 0
    epoch = 0
    scaler = torch.cuda.amp.GradScaler()
    while es > 0:
        train_metrics = train_epoch(
            model, train_loader, loss_function, optimizer, scheduler, device, scaler
        )
        epoch += 1
        eval_metrics = eval_epoch(model, val_loader, loss_function, device)
        # wandb.log({**train, **val})
        print("EPOCH", epoch)
        print(train_metrics)
        print(eval_metrics)
        ap = eval_metrics["Val AveragePrecision"]
        if ap > max_ap + eps:
            best_model = copy.deepcopy(model)
            es = early_stopping
            max_ap = ap
        else:
            es -= 1
    # wandb.finish()
    return best_model, max_ap


def tune_params_and_fit(
    model_object,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    param_space,
    device=torch.device("cpu"),
    early_stopping: int = 3,
    eps: int = 1e-3,
    max_evals: int = 5,
    timeout: int = 3600,
):
    trials = Trials()
    best = fmin(
        fn=partial(
            objective,
            model_object=model_object,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_function=loss_function,
            optimizer=None,
            scheduler=None,
            device=device,
            early_stopping=early_stopping,
        ),
        space=param_space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE),
        show_progressbar=True,
        timeout=timeout,
    )

    return best


def objective(
    params,
    model_object,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    device,
    early_stopping,
):
    model = model_object(hidden_dim=512, dropout=0.1, attn_heads=4, n_attn_layers=2)
    criterion = loss_function
    model = model.to(device)
    criterion = criterion.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=params['step_size'],
                                                gamma=params['gamma'])

    best_model, best_ap = train_model_early_stopping(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=CUDA_DEV,
        early_stopping=early_stopping,
    )
    torch.save(best_model.state_dict(), f"model_{best_ap}.pt")
    with open(f"params_{best_ap}.json", "w") as json_file:
        json.dump(params, json_file, cls=NpEncoder)
    return -best_ap


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
