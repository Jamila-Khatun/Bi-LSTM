# Bi-LSTM
A deep learning model that classifies Bangla memes into 5 categories: Funny, Offensive, Awareness, Judgemental, and Political using Bidirectional LSTM.

## Features

- **Text Preprocessing**: Specialized cleaning for Bangla text
- **Class Imbalance Handling**: Oversampling + class weighting
- **Model Architecture**: Bidirectional LSTM with dropout
- **Evaluation**: 
  - Accuracy, Precision, Recall, F1-score
  - Confusion matrix visualization
  - 5-fold cross-validation

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow openpyxl imbalanced-learn
```

## Dataset Structure

Excel file with:
- **Column 1**: Meme text (Bangla)
- **Column 2**: Label (one of: Funny, Offensive, Awareness, Judgemental, Political)

Example:
```
"এইটা তো অনেক ফানি!", Funny
"রাজনীতি নিয়ে বিতর্ক", Political
```

## Usage

1. **Run in Google Colab**:
```python
from google.colab import files
uploaded = files.upload()  # Upload your dataset
```

2. **Key Parameters**:
```python
max_len = 100  # Maximum text length
epochs = 20    # Training iterations
batch_size = 64
```

3. **Model Training**:
```python
model.fit(X_train_pad, y_train_cat, 
          epochs=epochs,
          batch_size=batch_size,
          validation_split=0.1)
```

## Performance Metrics

| Class        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| Funny        | 0.66      | 0.37   | 0.47     |
| Offensive    | 0.89      | 0.97   | 0.93     |
| Awareness    | 1.00      | 0.91   | 0.95     |
| Judgemental  | 0.59      | 0.84   | 0.70     |
| Political    | 1.00      | 1.00   | 1.00     |

**Overall Accuracy**: 81.94%

## Directory Structure

```
bangla-meme-classifier/
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Preprocessed data
├── models/                 # Saved model files
├── notebooks/              # Jupyter/Colab notebooks
├── utils/
│   ├── preprocess.py       # Text cleaning functions
│   └── visualize.py        # Plotting functions
└── README.md
```

## Limitations

1. Requires minimum 50 samples per class
2. Accuracy drops for sarcastic/ambiguous memes
3. Performance depends on Bangla text quality

## Citation

If you use this in research, please cite:
```bibtex
@misc{banglameme2023,
  title={Bangla Meme Classification using BiLSTM},
  author={Your Name},
  year={2023},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourrepo}}
```

Key sections included:
1. Badges for quick info
2. Clear installation instructions
3. Dataset format specification
4. Performance metrics table
5. Modular directory structure
6. Limitations and license
7. Citation template
