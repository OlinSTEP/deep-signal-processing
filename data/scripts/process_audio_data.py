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


def map_markers_to_timestamps(markers, timestamps):
    """
    Creates a list mapping marker indicies to nearest timestamp index
    For marker with idx i, idx of nearest timestamp will be mapping[i]
    """
    closest_idxs = []
    m = t = 0

    # First recording *should* always be before first marker
    assert timestamps[0] < markers[0]

    while m < len(markers) and t < len(timestamps):
        if timestamps[t] < markers[m]:
            t += 1
        elif timestamps[t] > markers[m]:
            last = timestamps[t - 1]
            last_diff = markers[m] - last
            now = timestamps[t]
            now_diff = now - markers[m]
            closest = t if now_diff < last_diff else t - 1
            closest_idxs.append(closest)
            m += 1
        else:
            closest_idxs.append(t)
            m += 1
            t += 1

    while m < len(markers):
        closest_idxs.append(timestamps[-1])
        m += 1

    return closest_idxs


def get_trailing_number(s):
    m = re.search(r'\d+$', s)
    return int(m.group()) if m else None


def get_stream_idx(streams, stream_name):
    for i, stream in enumerate(streams):
        if stream["info"]["name"][0] == stream_name:
            return i
    print(f"Stream {stream_name} not found!")
    get_names_exit(streams)


def get_names_exit(streams):
    for i, stream in enumerate(streams):
        print(f"{i}: {stream['info']['name'][0]}")
    sys.exit()


def split_streams(marker_stream, data_streams):
    marker_series = marker_stream["time_series"]
    marker_stamps = marker_stream["time_stamps"]

    markers_to_streams = {
        key: map_markers_to_timestamps(marker_stamps, stream["time_stamps"])
        for key, stream in data_streams.items()
    }

    all_data = []
    for i in range(len(marker_series) - 1):
        # Marks are in the form [b|e][target_idx][N|S][sequence_idx]
        marker_text = marker_series[i][0]
        marker_text_next = marker_series[i + 1][0]

        # Skip ending markers
        if marker_text[0] == 'e':
            continue

        # Check that this marker and the next one are a pair, otherwise skip
        if marker_text[0] == 'b' and marker_text_next[0] == 'e':
            data_dict = {
                "target_idx": int(marker_text[1]),
                "type": marker_text[2]  # V or S, voiced or silent
            }
            for key, mapping in markers_to_streams.items():
                stream_data = data_streams[key]["time_series"]
                data_dict[key] = stream_data[mapping[i]:mapping[i + 1]]
            all_data.append(data_dict)

    return all_data


def load_streams(
    file_path, reg_name, throat_name, marker_name, get_names
):
    print(f"Loading {file_path}...")
    streams, _ = pyxdf.load_xdf(file_path)

    if get_names:
        get_names_exit()

    for i, stream in enumerate(streams):
        print(f"{i}: {stream['info']['name'][0]}")

    reg_channel = get_stream_idx(streams, reg_name)
    throat_channel = get_stream_idx(streams, throat_name)
    marker_channel = get_stream_idx(streams, marker_name)
    print(f"Reg channel: {reg_channel}")
    print(f"Throat channel: {throat_channel}")
    print(f"Marker channel: {marker_channel}")

    print("Processing...")
    all_data = split_streams(
        streams[marker_channel],
        {
            "reg": streams[reg_channel],
            "throat": streams[throat_channel]
        }
    )
    sample_rates = {
        "reg": streams[reg_channel]["info"]["nominal_srate"][0],
        "throat": streams[throat_channel]["info"]["nominal_srate"][0],
    }

    return all_data, sample_rates


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


def save_datapoint(save_dir, i, datapoint, sample_rates):
    print(f"Writing {os.path.join(save_dir, str(i))}...")

    if len(datapoint["throat"]) / int(float(sample_rates["throat"])) < 0.4:
        print(f"Sample too short! Skipped. Label: {datapoint['target']}")
        return;

    file_path = os.path.join(save_dir, f"{i}_info.json")
    save_dict = {
        "target": datapoint["target"],
        "type": datapoint["type"]
    }
    with open(file_path, "w") as f:
        json.dump(save_dict, f)
    print(f"  Wrote {file_path}")

    reg_array = datapoint["reg"]
    reg_sample_rate = sample_rates["reg"]
    file_path = os.path.join(save_dir, f"{i}_reg_audio.wav")
    scipy.io.wavfile.write(
        file_path,
        int(float(reg_sample_rate)),
        reg_array
    )
    print(f"  Wrote {file_path}")

    throat_array = datapoint["throat"]
    throat_sample_rate = sample_rates["throat"]
    file_path = os.path.join(save_dir, f"{i}_throat_audio.wav")
    scipy.io.wavfile.write(
        file_path,
        int(float(throat_sample_rate)),
        throat_array
    )
    print(f"  Wrote {file_path}")


def save_datapoints(save_dir, all_data, sample_rates):
    os.makedirs(save_dir, exist_ok=True)
    for i, data_dict in enumerate(all_data):
        save_datapoint(save_dir, i, data_dict, sample_rates)


def process_session_dir(args, session_dir):
    save_dir = args.save
    reg_channel = args.reg
    throat_channel = args.throat
    marker_channel = args.marker
    get_names = args.get_names

    idx_to_target = {}
    for file_name in os.listdir(session_dir):
        path = os.path.join(session_dir, file_name)
        if file_name.endswith(".xdf"):
            all_data, sample_rates = load_streams(
                path, reg_channel, throat_channel, marker_channel, get_names
            )
        elif file_name.endswith(".json"):
            idx, target = load_targets(path)
            idx_to_target[idx] = target

    # Convert target_idx to target
    all_data = [
        {
            "reg": d["reg"],
            "throat": d["throat"],
            "target": idx_to_target[d["target_idx"]],
            "type": d["type"],
        } for d in all_data
    ]

    # Remove samples with no target
    all_data = [d for d in all_data if d["target"]]

    session_save_dir = os.path.join(save_dir, os.path.basename(session_dir))
    save_datapoints(session_save_dir, all_data, sample_rates)


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
    parser.add_argument(
        '--reg',
        type=str,
        default="AudioCaptureWin"
    )
    parser.add_argument(
        '--throat',
        type=str,
        default="MyAudioStream"
    )
    parser.add_argument(
        '--marker',
        type=str,
        default="MarkersForBooks"
    )
    parser.add_argument(
        '--get_names',
        default=False,
        action="store_true"
    )
    parser.add_argument(
        '--no_thread',
        default=False,
        action="store_true"
    )

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    os.makedirs(args.save, exist_ok=True)
    session_dirs = [os.path.join(args.data, n) for n in os.listdir(args.data)]

    if not args.no_thread:
        pool = Pool(processes=cpu_count())
        func = partial(process_session_dir, args)
        pool.map(func, session_dirs)
        pool.close()
        pool.join()
    else:
        for session_dir in session_dirs:
            process_session_dir(args, session_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
