# DNA Sequence Classification

Bioinformatics project combining Naïve String Matching (DA-3) with Machine Learning-based DNA classification (DA-4), featuring a gesture-controlled air-keyboard input interface.

## About

This project was developed as part of the Design and Analysis of Algorithms lab (VIT). It evolves across two stages:

- **DA-3**: Naïve string matching algorithm in C to detect gene patterns in DNA sequences, paired with a hand-tracking air-keyboard UI in Python for gesture-based sequence input
- **DA-4**: Machine learning enhancement — classifies DNA sequences as Promoter or Non-Promoter using two models: a Conv1D CNN with One-Hot encoding, and a Dense network with K-mer encoding

## Repository Structure

```
DNA-Sequence-Classification/
│
├── naive_match.c                  # DA-3: Naïve string matching in C
├── project.py                     # DA-3: Hand-tracking air-keyboard UI
├── da4_dna_classification.py      # DA-4: ML classification model
└── README.md                      # Project documentation
```

## DA-3 — Naïve String Matching + Hand Tracking UI

### Algorithm
Slides a pattern window across a DNA sequence character by character.
- Time Complexity: O(n × m)
- Space Complexity: O(1)

### Air-Keyboard UI
Gesture-based DNA input using a webcam. Raise your index finger and hover over the virtual keyboard for 10 frames to select a nucleotide. Built with OpenCV and MediaPipe.

### How to Run DA-3

1. Compile the C program:
   ```
   gcc naive_match.c -o naive_match
   ```
2. Install Python dependencies:
   ```
   pip install opencv-python mediapipe numpy
   ```
3. Run the UI:
   ```
   python project.py
   ```
4. Enter your DNA sequence and gene pattern using hand gestures or keyboard, then press Enter

## DA-4 — DNA Sequence Classification (ML)

### Models

| Model | Encoding | Architecture | AUC Score |
|-------|----------|--------------|-----------|
| Model A | One-Hot (57×4) | Conv1D CNN | 0.8614 |
| Model B | K-mer (k=3, 64-dim) | Dense Network | 0.8961 |

### Outputs Generated
- Training vs Validation Accuracy curves
- Training vs Validation Loss curves
- ROC Curve with AUC Score
- Confusion Matrix heatmap
- Classification Report (Precision, Recall, F1)

### How to Run DA-4

1. Install dependencies:
   ```
   pip install tensorflow scikit-learn matplotlib seaborn numpy
   ```
2. Run the classification model:
   ```
   python da4_dna_classification.py
   ```

## Tech Stack

- Python 3
- C (GCC)
- TensorFlow / Keras
- scikit-learn
- OpenCV
- MediaPipe
- NumPy, Matplotlib, Seaborn

## Domain

Bioinformatics — DNA Pattern Detection & Classification
Application: Medical Genomics, Promoter Prediction, Disease Diagnostics

## Author

Yohan George
24BAI0356 — B.Tech CSE (AI & ML), VIT
