import torch


class Dataset(torch.utils.data.Dataset):
    """
    A general Dataset class.

    Primarily serves as a wrapper to call various data loading classes as a
    complete data pipeline. Requires all data loading class attributes to be set
    in classes extending this one, see src/data/__init__ for examples.
    """

    loader_cls = None
    filter_cls = None
    input_encoder_cls = None
    target_encoder_cls = None

    def __init__(self, config):
        if (
            self.loader_cls is None
            or self.filter_cls is None
            or self.input_encoder_cls is None
            or self.target_encoder_cls is None
        ):
            raise NotImplementedError(
                "Data loading classes not set! Use a class that extends Dataset"
            )

        self.loader = self.build_loader(config)
        self.filter = self.build_filter(config)
        self.input_encoder = self.build_input_encoder(config)
        self.target_encoder = self.build_target_encoder(config)

    def __len__(self):
        return self.loader.len

    def __getitem__(self, i):
        input_, target = self.loader.load(i)
        filtered_input = self.filter.filter(input_)
        processed_input = self.input_encoder.transform(filtered_input)
        processed_target = self.target_encoder.transform(target)

        return {
            # (time, num_channels)
            'input': processed_input,
            # (1)
            'target': processed_target
        }

    def build_loader(self, config):
        return self.loader_cls(config)

    def build_filter(self, config):
        return self.filter_cls(config)

    def build_input_encoder(self, config):
        # Generator so we don't have to load all inputs into memory
        # input_encoder can load all inputs at its discretion
        def generator():
            for i in range(len(self)):
                input_, _ = self.loader.load(i)
                yield input_

        input_encoder = self.input_encoder_cls(config)
        input_encoder.fit(generator())

        return input_encoder

    def build_target_encoder(self, config):
        datapoints = [self.loader.load(i) for i in range(len(self))]
        targets = [target for _, target in datapoints]
        target_encoder = self.target_encoder_cls(config)
        target_encoder.fit(targets)
        return target_encoder

    def split(self):
        train_idx, dev_idx, test_idx = self.loader.build_splits()
        train_set = torch.utils.data.Subset(self, train_idx)
        dev_set = torch.utils.data.Subset(self, dev_idx)
        test_set = torch.utils.data.Subset(self, test_idx)
        return train_set, dev_set, test_set

    @property
    def collate_fn(self):
        return self.input_encoder.collate_fn

    @property
    def input_dim(self):
        if getattr(self, "input_encoder"):
            return self.input_encoder.input_dim
        return None

    @property
    def target_dim(self):
        if getattr(self, "target_encoder"):
            return self.target_encoder.target_dim
        return None


if __name__ == "__main__":
    # DO TESTS HERE
    pass
