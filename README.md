# Embedded Deep Learning System for Respiratory Disease Detection through Audio Analysis

Official implementation of the paper: link.

**"Paper Title"**
Embedded Deep Learning System for Respiratory Disease Detection through Audio Analysis
Computer Methods and Programs in Biomedicine, 2026

---

## Overview

This repository contains the code and experiments used in the paper.

Main contributions:

* A Deep Learning framework for the detection of respiratory diseases on embedded systems.
* Real-time inference in portable and resource-constrained environments.
* Open and reproducible.
* Achieves the best specificity result among works with similar evaluation and training methods.


---

## Repository Structure

```
project/
│
├── configs/        # experiment configurations
├── data/           # dataset instructions
├── models/         # saved checkpoints
├── src/            # source code
├── scripts/        # training / evaluation scripts
├── experiments/    # logs and results
└── notebooks/      # analysis notebooks
```

---

## Requirements

Python 3.10

Main dependencies:

* TensorFlow / PyTorch
* NumPy
* Pandas
* Librosa
* Scikit-learn

Install dependencies:

```
pip install -r requirements.txt
```

---

## Dataset

Download the dataset from:

[ICBHI 2017 Challenge Original Dataset](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

We will get two files for every recording: raw data (.WAV file) and related annotation file. After downloading, place the files in:

```
data/raw/
```

The expected structure of the folder, as an example, for the recording `101_1b1_Al_sc_Meditron`:

```
data/
   raw/
      101_1b1_Al_sc_Meditron.wav
      101_1b1_Al_sc_Meditron.txt
```

---

## Training

To train the model:

```
python src/training/train.py --config configs/config.yaml
```

---

## Evaluation

To evaluate a trained model:

```
python src/evaluation/evaluate.py --checkpoint models/checkpoints/best_model
```

---

## Reproducing the Results in the Paper

To reproduce the main results:

```
bash scripts/reproduce_results.sh
```

This script will:

1. Train the model
2. Evaluate the model
3. Generate the figures used in the paper

---

## Pretrained Models

Pretrained weights are available in:

```
models/pretrained/
```

---

## Results

Example results:

| Model          | Accuracy | F1 Score |
| -------------- | -------- | -------- |
| Proposed Model | 0.91     | 0.89     |

---

## Citation

If you use this code, please cite:

```
@article{author2025paper,
  title={Paper Title},
  author={Author1, Author2},
  journal={Journal Name},
  year={2025}
}
```

---

## License

MIT License