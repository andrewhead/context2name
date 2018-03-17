# Context2Name Project

## Assumptions

Assumes you already have the raw data files in `raw/`.  Ask
the repo admin if you want access to them.

## Getting Started

Set up dependencies:

```bash
virtualenv -v python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

Pre-process the data:

```bash
# Create input and output vocabularies.
python build_vocabularies.py raw/training_processed.txt --show-progress

# Process data into CSVs with token IDs instead of text.
python process_data.py raw/training_processed.txt --show-progress

# Split the data into training and validation.
python split_data.py processed/training_processed.csv --show-progress

# Shuffle the training data into random order.  Note: this
# may only work on OSX.  Try using `shuf` on Unix systems.
gshuf processed/training.csv -o processed/training_shuffled.csv

# Process evaluation data into compatible format.
python process_data.py raw/evaluation_preprocessed.txt --show-progress
```

The `--show-progress` flag is optional.
