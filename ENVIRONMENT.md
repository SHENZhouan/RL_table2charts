# Table2Charts Environment

This document summarizes the environment needed to run the current code in this
repository. It is based on the code imports, README/docker notes, run scripts,
and the working local `.venv`.

## 1. Core Environment

Use this for the current Table2Charts training, RL, evaluation, single inference,
and Plotly corpus conversion scripts.

- OS: Linux
- Python: 3.10.12
- CUDA wheel target: CUDA 11.8 (`torch==2.2.2+cu118`)
- GPU: recommended for training/evaluation scripts that set `CUDA_VISIBLE_DEVICES`
- Working directory for model commands: `Table2Charts/`

Install:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-table2charts-cu118.txt
```

Core Python package list:

| Package | Version | Why it is needed |
| --- | --- | --- |
| `torch` | `2.2.2+cu118` | Model, DDP, training, inference |
| `torchvision` | `0.17.2+cu118` | Installed in the working environment with PyTorch |
| `numpy` | `1.26.4` | Data tensors, features, reward/Q-value logic |
| `scipy` | `1.15.3` | Numeric/statistical utilities |
| `scikit-learn` | `1.7.2` | Metrics in SFT/RL training and evaluation |
| `tensorboard` | `2.20.0` | `torch.utils.tensorboard.SummaryWriter` |
| `tqdm` | `4.67.3` | Progress bars in data/Q-value processing |
| `sortedcontainers` | `2.4.0` | Beam/search ranking structures |
| `pika` | `1.3.2` | RabbitMQ queue mode for distributed RL |
| `requests` | `2.33.1` | Utility/network dependency present in `.venv` |
| `pillow` | `12.2.0` | Image utility dependency present in `.venv` |

The local `.venv` also includes transitive packages such as `filelock`,
`fsspec`, `Jinja2`, `networkx`, CUDA wheel libraries, `protobuf`, `sympy`,
`triton`, `typing_extensions`, `urllib3`, and `Werkzeug`; these are normally
installed automatically with the packages above.

## 2. System Tools

Recommended command-line tools:

- `bash`
- `git`
- `tmux` for long training/evaluation jobs
- `wget` or `curl` for downloading public corpora
- NVIDIA driver compatible with CUDA 11.8 wheels

For multi-node RL without `--queue_mode=local`, a RabbitMQ server is also needed
because `reinforce/*.py` can connect through `pika`. The current local scripts
mostly use `--queue_mode=local`, so RabbitMQ is not required for those runs.

## 3. Optional Packages

Install these only if you need the older data analysis, human evaluation, web
server, or baseline utilities:

```bash
pip install pandas matplotlib seaborn editdistance dateparser imbalanced-learn jsonschema pyyaml flask flask-cors pyopenssl
```

Common optional package uses:

| Package | Used by |
| --- | --- |
| `pandas` | `Results/HumanEvaluation`, VizML-style scripts |
| `matplotlib`, `seaborn` | VizML notebooks/analysis |
| `editdistance` | VizML feature extraction |
| `dateparser` | VizML feature extraction |
| `imbalanced-learn` | VizML neural-network utilities |
| `jsonschema` | VizML/baseline validation utilities |
| `pyyaml`, `flask`, `flask-cors`, `pyopenssl` | legacy Data2VisRaw web/server scripts |

## 4. Apex Is Optional

The original Dockerfile installed NVIDIA Apex and the README examples pass
`--apex`. The current code imports Apex inside `try/except` blocks and runs
without it as long as `--apex` is not passed. The current run scripts do not pass
`--apex`, so Apex is not part of the required core environment.

If you want to reproduce the original mixed-precision stack exactly, install
Apex separately for your CUDA/PyTorch combination and then add `--apex` to the
training command.

## 5. Legacy Baseline Environments

Do not mix the legacy baselines into the Python 3.10 core environment unless you
really need them. They target older libraries and are better isolated.

### VizML Baseline

The repository contains `vizml/requirements.txt` and
`Baselines/VizML/requirements.txt`:

```text
cycler==0.10.0
editdistance==0.5.3
kiwisolver==1.1.0
matplotlib==3.0.3
numpy==1.16.3
pandas==0.24.2
pyparsing==2.4.0
python-dateutil==2.8.0
pytz==2019.1
scikit-learn==0.20.3
scipy==1.2.1
six==1.12.0
```

Those versions are legacy and generally fit Python 3.6/3.7 better than Python
3.10.

### Data2VisRaw Baseline

`Data/Data2VisRaw` is based on old TensorFlow `tf.contrib` code. Use a separate
legacy environment, typically Python 3.5/3.6 with TensorFlow 1.x, plus:

```text
pyyaml
flask
python-dateutil
matplotlib
pyopenssl
flask-cors
pyrouge
```

### Draco Baseline

`Baselines/test_draco.py` expects the external Draco project/package to be
installed separately. Follow the Draco repository setup for the solver/runtime.

## 6. Quick Verification

After installing the core environment:

```bash
source .venv/bin/activate
python - <<'PY'
import torch, numpy, scipy, sklearn, tensorboard, tqdm, sortedcontainers, pika
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("ok")
PY
```

For a lightweight code-path check from this repository:

```bash
cd Table2Charts
../.venv/bin/python single_inference.py \
  --df_path ../Data/Example/data/0.t0.DF.json \
  --emb_path ../Data/Example/embeddings/fasttext/0.EMB.json \
  --model_path ../Results/Models/best-excel.pt \
  --device cuda \
  --max_steps 200
```

Use `--device cpu` only for smoke tests; real training/evaluation should use GPU.
