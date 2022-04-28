import random

import torch

from .dataset import Dataset

class TargetPairDataset(torch.utils.data.Dataset):
    def __init__(self, config, source_set, target_set):
        self.seed = config.seed
        self.pos_pairs = config.pos_pairs
        self.neg_pairs = config.neg_pairs
        self.source_set = source_set
        self.target_set = target_set

        random.seed(self.seed)

        source_cache = {}
        positive_pairs, negative_pairs = [], []
        for t_idx in range(len(target_set)):
            target_positive_pairs, target_negative_pairs = [], []
            target_target = target_set[t_idx]["target"]

            s_idxs = list(range(len(source_set)))
            random.shuffle(s_idxs)

            for s_idx in range(len(source_set)):
                if s_idx not in source_cache:
                    source_cache[s_idx] = source_set[s_idx]["target"]
                source_target = source_cache[s_idx]

                if source_target == target_target:
                    target_positive_pairs.append((s_idx, t_idx, 1))
                else:
                    target_negative_pairs.append((s_idx, t_idx, 0))

            random.shuffle(target_positive_pairs)
            positive_pairs.extend(target_positive_pairs[:self.pos_pairs])
            random.shuffle(target_negative_pairs)
            negative_pairs.extend(target_negative_pairs[:self.neg_pairs])

        self.pairs = positive_pairs + negative_pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        source_idx, target_idx, is_pos = self.pairs[idx]
        source_data = self.source_set[source_idx]
        target_data = self.target_set[target_idx]

        return {
            "source": source_data,
            "target": target_data,
            "is_pos": is_pos
        }


class DomainAdaptionDataset():
    loader_cls = None
    input_encoder_cls = None
    target_encoder_cls = None

    def __init__(self, config):
        if (
            self.loader_cls is None
            or self.input_encoder_cls is None
            or self.target_encoder_cls is None
        ):
            raise NotImplementedError(
                "Data loading classes not set! Use a class that extends"
                " DomainAdaptionDataset"
            )

        class InnerDataset(Dataset):
            loader_cls = self.loader_cls
            input_encoder_cls = self.input_encoder_cls
            target_encoder_cls = self.target_encoder_cls
        self.dataset = InnerDataset(config)

        subsets = self.dataset.split()
        if len(subsets) != 5:
            raise ValueError(
                f"Domain adaption expects 5 splits, received {len(subsets)}"
            )

        train, dev_train, dev_test, test_train, test_test = subsets
        self.dev_test = dev_test
        self.test_test = test_test
        self.dev_train = TargetPairDataset(config, train, dev_train)
        self.test_train = TargetPairDataset(config, train, test_train)

    def __len__(self, _):
        raise NotImplementedError(
            "DomainAdaptionDataset does not implement standard dataset"
            " functions. Only use subsets returned by split()"
        )

    def __getitem__(self, _):
        raise NotImplementedError(
            "DomainAdaptionDataset does not implement standard dataset"
            " functions. Only use subsets returned by split()"
        )

    def split(self):
        return self.dev_train, self.test_train, self.dev_test, self.test_test

    def collate_pairs(self, batch):
        source_batch, target_batch, is_pos = [], [], []
        for d in batch:
            source_batch.append(d["source"])
            target_batch.append(d["target"])
            is_pos.append(d["is_pos"])
        source_batch = self.dataset.collate_fn(source_batch)
        target_batch = self.dataset.collate_fn(target_batch)
        is_pos = torch.tensor(is_pos, dtype=torch.float32)
        return {
            "source_input": source_batch["input"],
            "source_target": source_batch["target"],
            "target_input": target_batch["input"],
            "target_target": target_batch["target"],
            "is_pos": is_pos
        }

    def collate_fn(self, batch):
        if "source" in batch[0]:
            return self.collate_pairs(batch)
        return self.dataset.collate_fn(batch)

    @property
    def input_dim(self):
        return self.dataset.input_dim

    @property
    def target_dim(self):
        return self.dataset.target_dim


if __name__ == "__main__":
    pass
