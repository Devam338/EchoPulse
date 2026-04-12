# EchoPulse

This implementation is based on the uploaded project report describing:
- audio-based diagnosis using **MFCC features**
- **data augmentation** with added noise and pitch shifting
- **Random Forest** and **SVM** models
- a combined prediction workflow for respiratory/cardiopulmonary 

## What this repo includes

- configurable data loading from WAV audio files
- MFCC-based feature extraction
- optional audio augmentation
- training pipelines for:
  - Random Forest
  - SVM
  - soft-voting ensemble
- evaluation with accuracy, classification report, and confusion matrix
- inference script for single audio files

## Dataset layout

Place your audio files in:  

```text
data/
  raw/
    asthma/
      sample_001.wav
    copd/
      sample_002.wav
    heart_failure/
      sample_003.wav
    normal/
      sample_004.wav
```


## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

Train models:

```bash
python scripts/train.py --data_dir data/raw --model all
```

Run inference:

```bash
python scripts/predict.py --audio_path data/raw/asthma/sample_001.wav --bundle_path models/model_bundle.joblib
```


## Streamlit app

After training a model bundle, you can launch a simple web interface:

```bash
streamlit run streamlit_app.py
```

The app lets you:
- upload an audio recording
- choose Random Forest, SVM, or Ensemble inference
- view the predicted label
- inspect class probabilities
- visualize the waveform and spectrogram
