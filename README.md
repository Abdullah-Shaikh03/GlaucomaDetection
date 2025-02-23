# Glaucoma Detection Model

This repository contains the implementation of a deep learning-based glaucoma detection model using the HybridNet architecture.

## Overview

The model is designed to classify fundus images as either **Glaucoma** or **Non-Glaucoma** using deep learning techniques. The training process, data augmentation, loss functions, and evaluation metrics are carefully designed to optimize performance.

## Model Performance

- **Best Validation Accuracy:** 93.51%
- **Test Accuracy:** 90.13%
- **Test Loss:** 0.2358

### Classification Report

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Non-Glaucoma | 0.93      | 0.86   | 0.90     | 385     |
| Glaucoma     | 0.87      | 0.94   | 0.90     | 385     |

- **Overall Accuracy:** 90.13%
- **Macro Average F1-Score:** 0.90
- **Weighted Average F1-Score:** 0.90

### Confusion Matrix

```
[[333  52]
 [ 24 361]]
```

## Training History

The training progress is visualized in `training_history.png`, showing accuracy and loss over epochs.

## Repository Structure

```
GlaucomaDetection/
├── Analysis_Results/
├── logs/
│   ├── run_001/
│   ├── run_xxx/
├── models/
├── Scripts/
├── raw/
│   ├── train/
│   │   ├── RG/
│   │   ├── NRG/
│   ├── test/
│   │   ├── RG/
│   │   ├── NRG/
│   ├── validation/
│   │   ├── RG/
│   │   ├── NRG/
├── .gitignore
├── LICENSE
└── README.md
```

## Requirements

To run the project, install the dependencies:

```bash
pip install -r requirements.txt
```

## Training the Model

Run the following command to start training:

```bash
python Scripts/Train.py
```

## Evaluation

To evaluate the trained model on test data:

```bash
python Scripts/visualize.py
```

## Contributors

- **Your Name** - [Your Email or GitHub](https://github.com/yourgithub)

## License

This project is licensed under the MIT License - see the LICENSE file for details.


<!-- export HSA_OVERRIDE_GFX_VERSION=10.3.0 -->

