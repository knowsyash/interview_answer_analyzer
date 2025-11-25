# Git Repository Cleanup Instructions

## Problem
Your `.git` folder is **1.56 GB** because large files (datasets, models, logs, kaggle.json) were previously committed to Git history.

## Solution
I've created a `.gitignore` file to prevent future tracking of large files. However, to actually reduce the repository size, you need to remove these files from Git history.

## Steps to Clean Up Git History

### Option 1: Using BFG Repo-Cleaner (Recommended - Fastest)

1. **Download BFG Repo-Cleaner:**
   ```powershell
   # Download from https://rtyley.github.io/bfg-repo-cleaner/
   # Or use Chocolatey:
   choco install bfg-repo-cleaner
   ```

2. **Make a backup:**
   ```powershell
   cd "f:\interview chat bot\"
   Copy-Item -Path "AI_Powered_Interview_Coach_Bot-_for_Job_Preparation" -Destination "AI_Powered_Interview_Coach_Bot-_BACKUP" -Recurse
   ```

3. **Clean the repository:**
   ```powershell
   cd "f:\interview chat bot\AI_Powered_Interview_Coach_Bot-_for_Job_Preparation"
   
   # Remove files larger than 1MB from history
   java -jar bfg.jar --strip-blobs-bigger-than 1M .
   
   # Or target specific files
   java -jar bfg.jar --delete-files kaggle.json .
   java -jar bfg.jar --delete-files "*.csv" .
   java -jar bfg.jar --delete-files "*.log" .
   java -jar bfg.jar --delete-files "*.joblib" .
   
   # Clean up
   git reflog expire --expire=now --all
   git gc --prune=now --aggressive
   ```

### Option 2: Using git filter-repo (More Control)

1. **Install git-filter-repo:**
   ```powershell
   pip install git-filter-repo
   ```

2. **Make a backup:**
   ```powershell
   cd "f:\interview chat bot\"
   Copy-Item -Path "AI_Powered_Interview_Coach_Bot-_for_Job_Preparation" -Destination "AI_Powered_Interview_Coach_Bot-_BACKUP" -Recurse
   ```

3. **Clean the repository:**
   ```powershell
   cd "f:\interview chat bot\AI_Powered_Interview_Coach_Bot-_for_Job_Preparation"
   
   # Remove specific files from history
   git filter-repo --invert-paths --path kaggle.json
   git filter-repo --invert-paths --path logs/session_log.txt
   git filter-repo --strip-blobs-bigger-than 1M
   ```

### Option 3: Manual git filter-branch (Built-in but Slower)

```powershell
cd "f:\interview chat bot\AI_Powered_Interview_Coach_Bot-_for_Job_Preparation"

# Remove specific files
git filter-branch --force --index-filter `
  "git rm --cached --ignore-unmatch kaggle.json logs/session_log.txt" `
  --prune-empty --tag-name-filter cat -- --all

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

## After Cleanup

1. **Check the new size:**
   ```powershell
   Get-ChildItem -Path ".git" -Recurse -File | Measure-Object -Property Length -Sum | Select-Object @{Name="SizeMB";Expression={[math]::Round($_.Sum / 1MB, 2)}}
   ```

2. **Force push to remote (if already pushed):**
   ```powershell
   git push origin --force --all
   git push origin --force --tags
   ```
   
   ⚠️ **Warning:** Force pushing rewrites history. Coordinate with team members if this is a shared repository.

## Files Now Ignored by .gitignore

- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- Jupyter checkpoints
- ML models (`*.pkl`, `*.joblib`, `*.h5`, `*.pth`)
- Large datasets (`*.csv`, `*.json`, `real_dataset_score/`)
- Kaggle credentials (`kaggle.json`)
- Logs (`logs/`, `*.log`)
- IDE settings (`.vscode/`, `.idea/`)

## Best Practices Going Forward

1. **Never commit large datasets** - use Git LFS or store externally
2. **Never commit credentials** - kaggle.json should stay local
3. **Never commit model files** - regenerate or download separately
4. **Keep logs local** - use logging services or local storage
5. **Document data sources** - add a README explaining where to get datasets

## Need Help?
If you encounter issues, the backup is at:
`f:\interview chat bot\AI_Powered_Interview_Coach_Bot-_BACKUP`
