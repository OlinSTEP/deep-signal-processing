# Deep Signal Processing

Signal processing library for using deep learning to process audio and gesture data.

# Setup

## Installing Dependencies

Setup venv:

```
python -m venv venv
```

Activate venv:

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```
## Running Notebooks

Install [Jupyter](https://jupyter.org/install)

Activate `ipywidgets` extension:

```
jupyter nbextension enable --py widgetsnbextension
```

After opening notebook, go to `Kernel` -> `Change Kernel` -> `env`

# Training a Model

A new model can be trained using the `train` script:

```
python3 -m src.runtimes.train --config=configs/audio_config.json --data data/processed_data/{DATA_DIR_NAME_HERE} --wandb_off
```

To upload data to Weights and Biases, run without the `--wandb_off` flag. Please note that the default WandB entity is `step-emg`, and the default WandB project is `Audio Signal Processing`. Both of these values can be tweaked in configs or with the appropriate command line arguments.

To save the model, specify `--save_dir`. By default, all models will be saved to `data/models/tmp`.

# Running / Evaluating a Trained Model

To test a model, run the `test` script:

```
python3 -m src.runtimes.test --load_dir data/models/{MODEL_DIR_HERE}/model
```

The original config is saved alongside the model, and is loaded for testing. Any previous values can be overwritten with new config file (specified with `--config`) or with command line arguments.

To demo a mode, run the `demo` script:

```
python3 -m src.runtimes.demo --load_dir data/models/{MODEL_DIR_HERE}/model
```

The user will be asked to hit `<enter>` to start and stop recording, after which a prediction and confidences will be printed.
