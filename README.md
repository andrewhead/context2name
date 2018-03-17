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
python build_vocabularies.py raw/training_preprocessed.txt --show-progress
python process_data.py raw/training_preprocessed.txt --show-progress
python process_data.py raw/evaluation_preprocessed.txt --show-progress
```

The `--show-progress` flag is optional.
