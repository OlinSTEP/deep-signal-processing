import io
import sys

import torch
import soundfile as sf
from flask import Flask, request

from src.config import config_from_args
from src.utils.save import load


app = Flask(__name__)
model = None
dataset = None

@app.route("/", methods=["POST"])
def process_image():
    if "audio_data" not in request.files:
        return "audio_data file not found", 400


    file = request.files['audio_data']
    data, samplerate = sf.read(io.BytesIO(file.read()))
    input_data = [
        (0, []),
        (0, []),
        (samplerate, data[:, 0]),
        (samplerate, data[:, 1]),
    ]
    processed_data = dataset.input_encoder.transform(input_data, False)


    out = model(processed_data[None, :])
    _, pred = torch.max(out, dim=1)

    out_idx = int(pred.numpy()[0])
    return {'msg': 'success', "pred_idx": out_idx}

def main(args):
    config = config_from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config.load_dir is None:
        raise ValueError("load_dir must be specified for running the webservice")
    config, built_objs = load(args, config.load_dir, device)

    global dataset, model
    dataset = built_objs[0]
    model = built_objs[4]
    model.eval()

    app.run(host='0.0.0.0', debug=True)


if __name__ == "__main__":
    main(sys.argv[1:])
