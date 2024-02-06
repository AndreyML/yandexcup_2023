from torch.utils.data import Dataset
import torch
import numpy as np

from settings import NUM_TAGS


class TaggingDataset(Dataset):
    def __init__(
        self, df, testing=False, track_idx2embeds=None, num_tags=NUM_TAGS, max_len=256
    ):
        self.df = df
        self.testing = testing
        self.track_idx2embeds = track_idx2embeds
        self.num_tags = num_tags
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_idx = row.track
        embeds = self.track_idx2embeds[track_idx]
        embed_len = embeds.shape[0]
        padding_len = self.max_len - embed_len
        pad = np.zeros((padding_len, embeds.shape[1]))
        mask = [0] * padding_len + [1] * embed_len
        embeds = np.concatenate([pad, embeds])
        if self.testing:
            return track_idx, embeds, mask
        tags = [int(x) for x in row.tags.split(",")]
        target = np.zeros(self.num_tags)
        target[tags] = 1
        return (track_idx, embeds, target, mask)


class NewTaggingDataset(Dataset):
    def __init__(
        self,
        df,
        testing=False,
        track_idx2embeds=None,
        num_tags=NUM_TAGS,
        max_len=256,
        i=0,
    ):
        self.df = df
        self.i = i
        self.testing = testing
        self.track_idx2embeds = track_idx2embeds
        self.num_tags = num_tags
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_idx = row.track
        embeds = self.track_idx2embeds[track_idx]
        embed_len = embeds.shape[0]
        padding_len = self.max_len - embed_len
        pad = np.zeros((padding_len, embeds.shape[1]))
        mask = [0] * padding_len + [1] * embed_len
        embeds = np.concatenate([pad, embeds])
        if self.testing:
            return track_idx, embeds, mask
        tags = [int(x) for x in row.tags.split(",")]
        target = np.zeros(self.num_tags)
        target[tags] = 1
        target = target[32 * self.i:32 * (self.i + 1)]
        return (track_idx, embeds, target, mask)


def collate_fn_train(b):
    (idx, embs, targets, mask) = zip(*b)
    track_idxs = torch.tensor(idx, dtype=torch.long)
    embeds = torch.from_numpy(np.array(embs)).float()
    targets = torch.from_numpy(np.array(targets)).float()
    mask = torch.tensor(mask, dtype=torch.long)
    return {"ids": track_idxs, "items": embeds, "labels": targets, "mask": mask}


def collate_fn_test(b):
    idx, embs, mask = zip(*b)
    track_idxs = torch.tensor(idx, dtype=torch.float32)
    embeds = torch.from_numpy(np.array(embs)).float()
    mask = torch.tensor(mask, dtype=torch.long)
    return {"ids": track_idxs, "items": embeds, "mask": mask}
