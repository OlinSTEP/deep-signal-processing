import os
import sys
import json
import queue
import threading

import scipy
import numpy as np
import sounddevice as sd

from src.config import config_from_args


class KeyboardThread(threading.Thread):
    def __init__(self, input_cbk=None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name)
        self.start()

    def run(self):
        self.input_cbk(input())  # Waits to get input + Return
        return

def record(target, samplerate=48000, device=0, channels=2):
    q = queue.Queue()
    b = queue.Queue()

    def unblock(_):
        b.put(1)

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        q.put(indata.copy())

    print()
    print(f"Next word is: {target}")
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


def main(args):
    config = config_from_args(args)
    save_dir = config.save_dir
    os.makedirs(save_dir, exist_ok=True)

    idx = 0
    samplerate = 48000
    labels = ["up", "down", "select", "back", "repeat"]
    while True:
        label = labels[idx % len(labels)]
        data = record(label, samplerate=samplerate)

        print(f"Writing {os.path.join(save_dir, str(idx))}...")

        file_path = os.path.join(save_dir, f"{idx}_info.json")
        save_dict = {
            "target": label
        }
        with open(file_path, "w") as f:
            json.dump(save_dict, f)
        print(f"  Wrote {file_path}")

        # Copy single input channel to two input channels
        throat_data = np.stack((data, data), axis=-1)

        # Only have throat data, copy for reg data to make loader happy
        file_path = os.path.join(save_dir, f"{idx}_reg_audio.wav")
        scipy.io.wavfile.write(
            file_path,
            samplerate,
            throat_data,
        )
        print(f"  Wrote {file_path}")

        file_path = os.path.join(save_dir, f"{idx}_throat_audio.wav")
        scipy.io.wavfile.write(
            file_path,
            samplerate,
            throat_data
        )
        print(f"  Wrote {file_path}")

        idx += 1


if __name__ == "__main__":
    main(sys.argv[1:])
