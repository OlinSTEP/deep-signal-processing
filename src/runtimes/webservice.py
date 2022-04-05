import io
import sys
import base64

import torch
import scipy
from flask import Flask, request

from src.config import config_from_args
from src.utils.save import load


app = Flask(__name__)
model = None
dataset = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.route("/", methods=["POST"])
def process_image():
    if "audio" not in request.json:
        return "audio not found", 400
    audio = request.json['audio']
    audio_decoded = base64.b64decode(audio)
    samplerate, data = scipy.io.wavfile.read(io.BytesIO(audio_decoded))

    input_data = [
        (0, []),
        (0, []),
        (samplerate, data),
        (samplerate, data),
    ]
    processed_data = dataset.input_encoder.transform(input_data, False).to(device)


    out = model(processed_data[None, :])
    _, pred = torch.max(out, dim=1)

    out_idx = int(pred.cpu().numpy()[0])
    print(dataset.target_encoder.inverse_transform(out_idx))
    return {'msg': 'success', "pred_idx": out_idx}

def main(args):
    config = config_from_args(args)

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
