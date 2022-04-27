import os
import re
import sys
import argparse
import shutil

import sox
from tqdm import tqdm


subjects = {}


class Subject():
    def __init__(self, idx):
        self.idx = idx
        self.sessions = {}


class Session():
    def __init__(self, idx):
        self.idx = idx
        self.len = 0


def parse_file(path):
    base_name = os.path.basename(path)
    match = re.match(r'^(\w+)(\d+)_(\d+)_(\w+)_(.+)\.(json|wav)', base_name)
    if match is None:
        return None

    # Only need to parse the json files
    ext = match.group(6)
    if ext == "json":
        return None

    name = match.group(1)
    session = int(match.group(2))
    # idx = match.group(3)
    # label = match.group(4)
    # id_ = match.group(5)

    no_ext_path = os.path.splitext(path)[0]
    wav_path = no_ext_path + ".wav"
    json_path = no_ext_path + ".json"
    file_names = (wav_path, json_path)

    return name, session, file_names


tfm = sox.Transformer()
def save_file(save_dir, name, session_name, file_names):
    wav_load_path, json_load_path = file_names

    if name not in subjects:
        subjects[name] = Subject(len(subjects))
    subject = subjects[name]

    if session_name not in subject.sessions:
        subject.sessions[session_name] = Session(len(subject.sessions))
    session = subject.sessions[session_name]

    session_dir = os.path.join(
        save_dir,
        f"subject_{subject.idx}",
        f"session_{session.idx}"
    )
    os.makedirs(session_dir, exist_ok=True)

    wav_save_path = os.path.join(
        session_dir,
        f"{session.len}_throat_audio.wav"
    )
    json_save_path = os.path.join(
        session_dir,
        f"{session.len}_info.json"
    )

    tfm.build(wav_load_path, wav_save_path)
    shutil.copy(json_load_path, json_save_path)

    session.len += 1


def parse_args(args):
    parser = argparse.ArgumentParser('Process Audio Data')

    parser.add_argument(
        '--data',
        type=str,
        default="data/raw_data/firebase_data/data"
    )
    parser.add_argument(
        '--save',
        type=str,
        default="data/processed_data/multi_subject"
    )

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    paths = [
        os.path.join(args.data, n)
        for n in os.listdir(args.data)
    ]
    parsed_paths = [parse_file(path) for path in paths]
    parsed_paths = sorted([p for p in parsed_paths if p is not None])

    print("Processing...")
    for subject, session, file_names in tqdm(parsed_paths):
        save_file(args.save, subject, session, file_names)
    print("Done.")

    print({k: v.idx for k, v in subjects.items()})

if __name__ == "__main__":
    main(sys.argv[1:])
