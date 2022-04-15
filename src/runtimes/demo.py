import sys
import queue
import threading

import torch
import sounddevice as sd
import numpy as np
import scipy
import librosa.display

from src.config import config_from_args
from src.utils.save import load
from src.utils.build import build_device


class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name)
        self.start()

    def run(self):
        self.input_cbk(input())  # Waits to get input + Return
        return

def record(samplerate=48000, device=0, channels=2):
    q = queue.Queue()
    b = queue.Queue()

    def unblock(_):
        b.put(1)

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print()
    print('press <enter> to start the recording')
    input()
    kthread = KeyboardThread(unblock)
    with sd.InputStream(
        samplerate=samplerate,
        device=device,
        channels=channels,
        callback=callback,
        dtype=np.int16
    ):
        print('Recording! <enter> to stop the recording')
        while b.empty():
            pass

    kthread.join()
    data = []
    while not q.empty():
        data.append(q.get())
    data = np.concatenate(data, axis=0)
    return data


def main(args, graph=False):
    config = config_from_args(args)
    device = build_device()

    if config.load_dir is None:
        raise ValueError("load_dir must be specified for demo")
    config, built_objs = load(args, config.load_dir, device)
    dataset = built_objs[0]
    model = built_objs[4]

    model.eval();
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
        confidences = torch.nn.functional.softmax(out[0], dim=0)
        _, pred = torch.max(out, dim=1)
        out_idx = int(pred.numpy()[0])
        pred = dataset.target_encoder.target_labels[out_idx]

        print()
        print("#" * 70)
        print(f"Predicted label: {pred}")
        print("Confidences:")
        for i, c in enumerate(confidences.detach().cpu().numpy()):
            print(f"  {dataset.target_encoder.target_labels[i]}: {c:.3f}")
        print("#" * 70)

        if graph:
            print(config.samplerate)
            print(config.max_ms)
            print(config.loudness)
            print(processed_data.shape)
            plot_data = processed_data.detach().cpu().numpy()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(
                plot_data[0, :, :], y_axis="mel", x_axis="time"
            )
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram')
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])
