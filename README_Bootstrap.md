# Bootstrap Model Setup

This script creates initial EEG model files to eliminate Streamlit warnings when starting the application.

## Prerequisites

1. Create a Python virtual environment:
   ```bash
   cd /Users/chenxiaomin/neuroplan
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

3. Install required packages:
   ```bash
   pip install numpy scikit-learn streamlit
   ```

## Usage

Run the bootstrap script to generate initial model files:

```bash
cd /Users/chenxiaomin/neuroplan
source venv/bin/activate
python src/app/eeg/bootstrap_model.py
```

## What it does

- Generates dummy EEG data (100 samples with 20 features each)
- Creates synthetic labels (relaxed=0, focused=1)  
- Trains a LinearSVC model with PCA preprocessing
- Saves the trained model and metadata to `model_store/eeg_mem_model/`

## Output

After running successfully, you should see:
- `model_store/eeg_mem_model/eeg_mem_model.model` - The trained model (pickle file)
- `model_store/eeg_mem_model/eeg_mem_model.json` - Model metadata (JSON file)

## Notes

- This creates a "dummy" model just to prevent Streamlit warnings
- The model is trained on synthetic data, not real EEG data
- Replace this model with a properly trained model using real data for production use
- The warnings during training about PCA components are expected with dummy data

## Next Steps

Once the initial model is created:
1. Your Streamlit app should load without model-related warnings
2. Train a proper model using real EEG data when available
3. Replace the dummy model files with the real trained model