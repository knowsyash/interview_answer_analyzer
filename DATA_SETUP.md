# Data Setup Guide

## Required Data Files (Not in Git)

This project requires several data files that are **NOT** included in the repository due to their large size. You need to set them up locally.

## Directory Structure

Create the following directories:

```
AI_Interview_Bot/data/real_dataset_score/
Research_Analysis/data/real_dataset_score/
Research_Analysis/outputs/transformer_model/
```

## Required Files

### 1. Training Datasets

Place these CSV files in `AI_Interview_Bot/data/real_dataset_score/`:

- `interview_data_with_scores.csv` - Behavioral interview questions (1,470 samples)
- `stackoverflow_training_data.csv` - Technical Q&A from Stack Overflow (10,000 samples)
- `webdev_interview_qa.csv` - Web development interview questions (44 samples)
- `combined_training_data.csv` - Combined dataset (11,514 samples)

**Source:** Generate using the Jupyter notebook `Research_Analysis/Optimized_Model_Training.ipynb`

### 2. Trained Models

Place these model files in `Research_Analysis/data/real_dataset_score/`:

- `optimized_ensemble_model.joblib` - Trained ensemble model (~86 MB)
- `model_config.joblib` - Model configuration

**Source:** Train using `Research_Analysis/Optimized_Model_Training.ipynb`

### 3. Transformer Model (Optional)

If using transformer models, place in `Research_Analysis/outputs/transformer_model/`:

- `pytorch_model.bin` - PyTorch transformer model (~268 MB)
- Other model files as needed

### 4. Kaggle API Key

Place your `kaggle.json` in the project root for dataset downloads.

**Get it from:** https://www.kaggle.com/settings → API → Create New API Token

## Quick Setup Commands

```powershell
# Create directories
New-Item -ItemType Directory -Force -Path "AI_Interview_Bot/data/real_dataset_score"
New-Item -ItemType Directory -Force -Path "Research_Analysis/data/real_dataset_score"
New-Item -ItemType Directory -Force -Path "Research_Analysis/outputs/transformer_model"

# Copy your kaggle.json (get it from Kaggle website first)
# It should contain: {"username":"your_username","key":"your_api_key"}
Copy-Item "path/to/your/kaggle.json" -Destination "kaggle.json"
```

## Generate Data

Run the Jupyter notebook to generate datasets and train models:

```powershell
jupyter notebook Research_Analysis/Optimized_Model_Training.ipynb
```

Or run individual cells to:
1. Download/load source datasets
2. Process and combine data
3. Train the ensemble model
4. Save models to the appropriate directories

## Why These Files Are Excluded

- **Size:** Total ~363 MB of models and datasets
- **Privacy:** kaggle.json contains API credentials
- **Reproducibility:** Models should be retrained with your data
- **Version Control:** Binary files don't diff well in Git

## Notes

- All these paths are in `.gitignore` and won't be committed
- Models can be regenerated from the training notebook
- Datasets can be downloaded from Kaggle or other sources
- Keep `kaggle.json` secret - never commit it!
