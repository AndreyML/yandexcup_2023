import numpy as np
import torch
from tqdm.autonotebook import tqdm

from settings import CUDA_DEV


def predict(model, loader):
    model.eval()
    track_idxs = []
    preds = []
    for batch in tqdm(loader):
        for key in batch:
            batch[key] = batch[key].to(CUDA_DEV)

        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits)
            preds.append(probs.cpu().numpy())
            track_idxs.append(batch["ids"].cpu().numpy())
    preds = np.concatenate(preds)
    track_idxs = np.concatenate(track_idxs)
    return track_idxs, preds
