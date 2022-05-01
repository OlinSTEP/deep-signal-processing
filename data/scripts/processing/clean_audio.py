import os
import re
import sys
import json
import argparse

from collections import defaultdict

import cleanlab
import torchaudio
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from speechbrain.pretrained import EncoderClassifier

from tqdm import tqdm


SEED = 42
lab2idx = {"back": 0, "down": 1, "repeat": 2, "select": 3, "up": 4}


def parse_file(path):
    base_name = os.path.basename(path)
    match = re.match(r'^(\D+)(\d+)_(\d+)_(\w+)_(.+)\.(json|wav)', base_name)
    if match is None:
        return None

    # Only need to parse the json files
    ext = match.group(6)
    if ext == "json":
        return None

    no_ext_path = os.path.splitext(path)[0]
    wav_path = no_ext_path + ".wav"
    json_path = no_ext_path + ".json"

    with open(json_path, 'r') as f:
        target = lab2idx[json.load(f)["target"]]

    return wav_path, target


def get_embeddings(paths):
    model = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-xvect-voxceleb"
        # "speechbrain/google_speech_command_xvector"
    )
    embeddings_list = []
    print("Producing embeddings...")
    for file_name in tqdm(paths):
        signal, _ = torchaudio.load(file_name)
        embeddings = model.encode_batch(signal)
        embeddings_list.append(embeddings.cpu().numpy())
    embeddings_array = np.squeeze(np.array(embeddings_list))
    return embeddings_array


def get_probs(embeddings, targets):
    print("Training linear regresser...")
    model = LogisticRegression(
        C=0.01,
        max_iter=10000,
        tol=1e-1,
        random_state=SEED
    )

    num_crossval_folds = 5
    cv_pred_probs = cross_val_predict(
        estimator=model,
        X=embeddings,
        y=targets,
        cv=num_crossval_folds,
        method="predict_proba"
    )
    predicted_labels = cv_pred_probs.argmax(axis=1)
    cv_accuracy = accuracy_score(targets, predicted_labels)
    print(f"CV Accuracy: {cv_accuracy}")
    return cv_pred_probs


def get_issue_idxs(probs, targets):
    print("Finding issue files...")
    label_issues_indices = cleanlab.filter.find_label_issues(
        labels=targets,
        pred_probs=probs,
        return_indices_ranked_by="self_confidence",
    )
    return label_issues_indices


def print_issue_idxs(paths, idxs):
    print(f"{len(idxs)} problematic files found!")
    input()

    subjects = defaultdict(list)
    issue_paths = [paths[i] for i in idxs]
    for ip in issue_paths:
        match = re.match(r'^(\D+)(\d+)_(\d+)_(\w+)_(.+)\.(json|wav)', ip)
        name = match.group(1)
        subjects[name].append(os.path.basename(ip))

    for name, ips in sorted(subjects.items()):
        print(f"Subject: {name} | Files: {len(ips)}")
        input()
        for ip in sorted(ips):
            print(f"  {ip}")
        input()


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
    paths, targets = list(zip(*parsed_paths))

    embeddings = get_embeddings(paths)
    probs = get_probs(embeddings, targets)
    idxs = get_issue_idxs(probs, targets)
    print_issue_idxs(paths, idxs)

if __name__ == "__main__":
    main(sys.argv[1:])
