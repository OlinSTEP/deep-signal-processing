import torch


class Dataset(torch.utils.data.Dataset):
    loader_cls = None
    filter_cls = None
    input_encoder_cls = None
    target_encoder_cls = None

    def __init__(self, config, dev=False, test=False):
        if (
            self.loader_cls is None
            or self.filter_cls is None
            or self.input_encoder_cls is None
            or self.target_encoder_cls is None
        ):
            raise NotImplementedError(
                "Data loading classes not set! Use a class that extends Dataset"
            )

        self.loader = self.build_loader(config, dev=dev, test=test)
        self.filter = self.build_filter(config)

        # Dev and test sets should not build encoders
        # Instead, set_encoding() should be used to take encoders from train set
        if not dev and not test:
            self.input_encoder = self.build_input_encoder(config)
            self.target_encoder = self.build_target_encoder(config)

    def __len__(self):
        return self.loader.len

    def __getitem__(self, i):
        if self.input_encoder is None or self.target_encoder is None:
            raise Exception(
                "For dev and test sets, set_encoding() must be called to copy "
                "encoders from the train set."
            )

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

    def build_loader(self, config, dev=False, test=False):
        return self.loader_cls(config, dev=dev, test=test)

    def build_filter(self, config):
        return self.filter_cls(config)

    def build_input_encoder(self, config):
        datapoints = [self.loader.load(i) for i in range(len(self))]
        inputs = [input_ for input_, _ in datapoints]
        input_encoder = self.input_encoder_cls(config)
        input_encoder.fit(inputs)
        return input_encoder

    def build_target_encoder(self, config):
        datapoints = [self.loader.load(i) for i in range(len(self))]
        targets = [target for _, target in datapoints]
        target_encoder = self.target_encoder_cls(config)
        target_encoder.fit(targets)
        return target_encoder

    def set_encoding(self, test_set):
        self.input_encoder = test_set.input_encoder
        self.target_encoder = test_set.target_encoder

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
    pass
