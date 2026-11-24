# ECS171GroupProject
Naive Bayes under label noise

## Setup

Use Python 3.14 and the exact versions in `requirements.txt`: the saved models in `models/` may not load with a different scikit-learn.

With [uv](https://docs.astral.sh/uv/) (installs Python 3.14 for you):

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Or, if Python 3.14 is already installed:

```
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dataset (`data/adult.data`, `data/adult.test`) is in the repo.

## Live demo

```
python server.py
```

Then open http://127.0.0.1:5000. The Flask server loads the trained models each notebook saves to `models/` (`save_model` in `preprocessing.py`) and runs them on the person shown in the page. The error plot comes from the test-set predictions each notebook saves to `results/` (`save_predictions`). Restart the server after re-running a notebook.
