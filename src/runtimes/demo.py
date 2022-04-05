import sys
import queue

import torch
import sounddevice as sd
import numpy as np

from src.config import config_from_args
from src.utils.save import load


def record(samplerate=48000, device=6, channels=1):
    """
    Primarily pulled from:
    https://python-sounddevice.readthedocs.io/en/0.4.4/examples.html#recording-with-arbitrary-duration
    """
    q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    try:
        print()
        print('press <enter> to start the recording')
        input()

        with sd.InputStream(samplerate=samplerate, device=device,
                            channels=channels, callback=callback):
            print('Recording! Press Ctrl+C to stop the recording')
            while True:
                pass
    except KeyboardInterrupt:
        data = []
        while not q.empty():
            data.append(q.get())
        data = np.concatenate(data, axis=0)
        return data


def main(args):
    config = config_from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.load_dir is None:
        raise ValueError("load_dir must be specified for demo")
    config, built_objs = load(args, config.load_dir, device)
    dataset = built_objs[0]
    model = built_objs[4]

    samplerate = 48000
    while True:
        data = record(samplerate=samplerate)
        input_data = [
            (0, []),
            (0, []),
            (samplerate, data[:, 0]),
            (samplerate, data[:, 0]),
        ]
        processed_data = dataset.input_encoder.transform(input_data, False)

        out = model(processed_data[None, :])
        _, pred = torch.max(out, dim=1)
        out_idx = int(pred.numpy()[0])
        pred = dataset.target_encoder.target_labels[out_idx]
        print()
        print("#" * 70)
        print(f"Predicted label: {pred}")
        print("#" * 70)


if __name__ == "__main__":
    main(sys.argv[1:])
