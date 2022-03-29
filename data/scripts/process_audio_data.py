import os
import re
import sys
import json
import argparse

from functools import partial
from multiprocessing import Pool, cpu_count

import pyxdf
import numpy as np
import scipy.io.wavfile


REG_CHANNEL = 0
THROAT_CHANNEL = 1
MARKER_CHANNEL = 2


def find_nearest_idx(array, value):
  array = np.asarray(array)
  return (np.abs(array - value)).argmin()


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def build_marker_list(streams):
    marker_list = []
    idx_list = []
    time_series = streams[MARKER_CHANNEL]["time_series"]
    time_stamps = streams[MARKER_CHANNEL]["time_stamps"]
    for i in np.arange(len(time_series) - 1):
      marker_text = time_series[i][0]
      marker_text_next = time_series[i + 1][0]

      # Check that this marker and the next one are a pair, otherwise skip
      if marker_text[0] == marker_text_next[0]:
        # Record which type it is and the starting and end time
        marker_list.append((
            marker_text[0],  # V or S, voiced or silent
            time_stamps[i],
            time_stamps[i+1]
        ))
        idx_list.append(get_trailing_number(marker_text))
    return marker_list, idx_list


def split_stream(stream, marker_list):
    split_data = []
    for (_, start, end) in marker_list:
        stream_start = find_nearest_idx(stream["time_stamps"], start)
        stream_end = find_nearest_idx(stream["time_stamps"], end)
        split_data.append(stream["time_series"][stream_start:stream_end])
    return split_data


def load_streams(file_path):
    print(f"Loading {file_path}...")
    streams, _ = pyxdf.load_xdf(file_path)

    print("Processing...")
    marker_list, idx_list = build_marker_list(streams)
    reg_data = split_stream(streams[REG_CHANNEL], marker_list)
    reg_sample_rate = streams[REG_CHANNEL]["info"]["nominal_srate"][0]
    throat_data = split_stream(streams[THROAT_CHANNEL], marker_list)
    throat_sample_rate = streams[THROAT_CHANNEL]["info"]["nominal_srate"][0]

    all_data = [
        {"reg": (r, reg_sample_rate), "throat": (t, throat_sample_rate), "target_idx": i}
        for r, t, i in zip(reg_data, throat_data, idx_list)
    ]

    return all_data


def load_targets(file_path):
    file_name = os.path.basename(file_path)
    match = re.match(r'(\d+)_info.json', file_name)
    if match is None:
        raise Exception(
            f"Improperly named file '{file_path}'. "
            "json files are expected to be in the format 'IDX_info.json'."
        )
    idx = int(match.group(1))

    with open(file_path) as f:
        data = json.load(f)
    target = data["text"]

    return idx, target


def save_datapoint(save_dir, i, datapoint):
    print(f"Writing {os.path.join(save_dir, str(i))}...")

    file_path = os.path.join(save_dir, f"{i}_info.json")
    save_dict = {"target": datapoint["target"]}
    with open(file_path, "w") as f:
        json.dump(save_dict, f)
    print(f"  Wrote {file_path}")

    reg_array, reg_sample_rate = datapoint["reg"]
    file_path = os.path.join(save_dir, f"{i}_reg_audio.wav")
    scipy.io.wavfile.write(
        file_path,
        int(reg_sample_rate),
        reg_array
    )
    print(f"  Wrote {file_path}")

    throat_array, throat_sample_rate = datapoint["throat"]
    file_path = os.path.join(save_dir, f"{i}_throat_audio.wav")
    scipy.io.wavfile.write(
        file_path,
        int(throat_sample_rate),
        throat_array
    )
    print(f"  Wrote {file_path}")


def save_datapoints(save_dir, all_data):
    os.makedirs(save_dir, exist_ok=True)
    for i, data_dict in enumerate(all_data):
        save_datapoint(save_dir, i, data_dict)


def process_session_dir(save_dir, session_dir):
    idx_to_target = {}
    for file_name in os.listdir(session_dir):
        path = os.path.join(session_dir, file_name)
        if file_name.endswith(".xdf"):
            all_data = load_streams(path)
        elif file_name.endswith(".json"):
            idx, target = load_targets(path)
            idx_to_target[idx] = target

    # Convert target_idx to target
    all_data = [
        {
            "reg": d["reg"],
            "throat": d["throat"],
            "target": idx_to_target[d["target_idx"]]
        } for d in all_data
    ]

    # Remove samples with no target
    all_data = [d for d in all_data if d["target"]]

    session_save_dir = os.path.join(save_dir, os.path.basename(session_dir))
    save_datapoints(session_save_dir, all_data)


def parse_args(args):
    parser = argparse.ArgumentParser('Process Audio Data')

    parser.add_argument(
        '--data',
        type=str,
        default="data/raw_data/SpringBreakAudio"
    )
    parser.add_argument(
        '--save',
        type=str,
        default="data/processed_data/SpringBreakAudio"
    )

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    os.makedirs(args.save, exist_ok=True)
    session_dirs = [os.path.join(args.data, n) for n in os.listdir(args.data)]

    pool = Pool(processes=cpu_count())
    func = partial(process_session_dir, args.save)
    pool.map(func, session_dirs)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main(sys.argv[1:])
