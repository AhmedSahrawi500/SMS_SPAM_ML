# SMS Spam ML (PyTorch + Streamlit)

A simple end-to-end SMS spam classifier using an LSTM model in PyTorch, with a Streamlit web app for interactive predictions.

## Features

- Train an LSTM spam classifier from CSV data
- Save/load model checkpoint (`spam_lstm_checkpoint.pt`)
- Predict from a web UI with adjustable spam threshold
- Shows prediction confidence and token coverage

## Project Structure

- `train.py` - trains the model and saves checkpoint
- `app.py` - Streamlit app for inference
- `spamraw.csv` - training data source
- `spam_lstm_checkpoint.pt` - saved model checkpoint
- `requirements.txt` - Python dependencies

## Requirements

- Python 3.10+ (tested with Python 3.12)
- pip
- Virtual environment (recommended)

## Setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data Format

`train.py` expects `spamraw.csv` with the columns:

- `type` - `ham` or `spam`
- `text` - message content

Example:

```csv
type,text
ham,Hey are we still meeting at 6?
spam,You won a free prize! Click now.
```

## Train the Model

```powershell
python train.py
```

Output:

- Creates/updates `spam_lstm_checkpoint.pt`
- Prints training/validation metrics per epoch

## Run the App

```powershell
python -m streamlit run app.py --server.port 0
```

Notes:

- `--server.port 0` lets Streamlit choose a free port automatically.
- If you use a fixed port (for example `8501`) and it is busy, choose another port.

## Usage

1. Open the local URL shown in terminal.
2. Paste or type an SMS message.
3. Adjust the spam threshold slider if needed.
4. Click **Predict**.

## Troubleshooting

### `Checkpoint not found`

If the app shows checkpoint missing, train first:

```powershell
python train.py
```

### `ModuleNotFoundError`

Install dependencies in the active environment:

```powershell
python -m pip install -r requirements.txt
```

### `Port is not available`

Run with auto-port selection:

```powershell
python -m streamlit run app.py --server.port 0
```

## License

Ahmed/Haiss
