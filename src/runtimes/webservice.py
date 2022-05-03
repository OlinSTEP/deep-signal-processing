import io
import sys
import base64

import torch
import scipy
import pyttsx3
from flask import Flask, request

from src.config import config_from_args
from src.utils.save import load
from src.utils.build import build_device


app = Flask(__name__)
model = None
dataset = None
device = build_device()
channels = 1
engine = pyttsx3.init()

@app.route("/", methods=["POST"])
def process_audio():
    if "audio" not in request.json:
        return "audio not found", 400
    audio = request.json['audio']
    audio_decoded = base64.b64decode(audio)
    samplerate, data = scipy.io.wavfile.read(io.BytesIO(audio_decoded))

    input_data = [(samplerate, data) for _ in range(channels)]
    processed_data = dataset.input_encoder.transform(input_data, False).to(device)

    with torch.no_grad():
        out = model(processed_data[None, :])
    out = torch.squeeze(out)
    _, pred = torch.max(out, dim=0)
    confidences = torch.nn.functional.softmax(out, dim=0)

    out_idx = int(pred.detach().cpu().numpy())
    target = dataset.target_encoder.inverse_transform(out_idx)
    print(target)
    print(list(confidences.detach().cpu().numpy()))

    engine.say(target)
    engine.runAndWait()

    return {'msg': 'success', "pred_idx": out_idx}

def main(args):
    config = config_from_args(args)

    if config.load_dir is None:
        raise ValueError("load_dir must be specified for running the webservice")
    config, built_objs = load(args, config.load_dir, device)

    global dataset, model, channels
    dataset, _, model, _, _ = built_objs
    channels = config.channels
    model.eval()

    app.run(host='0.0.0.0', debug=True)


if __name__ == "__main__":
    main(sys.argv[1:])
