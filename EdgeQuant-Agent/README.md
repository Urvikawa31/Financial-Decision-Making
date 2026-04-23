# 🏦 INVESTOR-AGENT: High-Conviction Hedge Fund PM

This repository implements an autonomous trading agent designed as a **High-Conviction Hedge Fund Portfolio Manager**. It specializes in capturing Alpha by identifying significant Catalyst Magnitude and Expectation Variance across multiple assets (BTC & TSLA).

---

## 🚀 Step-by-Step Setup Guide

Follow these steps to get the system running on your local machine.

### 1. Get Hugging Face Access
Since the system uses **Llama 3.3 70B**, you need a Hugging Face account and a token.

1.  **Apply for Model Access**: Go to [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and apply for access. Approval is usually instant.
2.  **Generate Token**:
    *   Log in to [Hugging Face](https://huggingface.co/).
    *   Go to **Settings** -> **Access Tokens**.
    *   Click **New Token**, name it (e.g., `InvestorAgent`), and set the type to **Read**.
    *   Copy the token; you will need it later.

### 2. Install Libraries
Ensure you have Python 3.10 or higher installed.

```bash
# 1. Clone the repository
# (Navigate to the project folder)

# 2. Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install Local Inference specialized libraries
# (Required for loading 70B models on local GPUs)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install accelerate bitsandbytes
```

### 3. Configure Authentication
There are two ways to provide your Hugging Face token to the system:

#### Option A: Using Hugging Face CLI (Recommended)
This stores your token globally on your machine.
```bash
# 1. Install the CLI
pip install huggingface_hub

# 2. Login (Paste your token when prompted)
huggingface-cli login
```

#### Option B: Using .env File
Alternatively, create a `.env` file in the root directory:
```bash
echo "HF_TOKEN=your_token_here" > .env
```

---

## 📈 Running the Pipeline

The agent operates in three distinct phases: Warmup, Test, and Evaluation.

### Phase 1: Warmup (Memory Building)
In this phase, the agent "learns" from historical data and populates its memory with reflections and patterns.
```bash
python run.py warmup
```
*   **What happens**: The agent processes data from `warmup_start_time` to `warmup_end_time`.
*   **Output**: Checkpoints are saved in `checkpoints/warmup`.

### Phase 2: Test (Trading Simulation)
The actual trading simulation where the agent makes decisions based on its built memory and new incoming news.
```bash
python run.py test
```
*   **What happens**: The agent processes data from `test_start_time` to `test_end_time`.
*   **Output**: Trading decisions and portfolio state are saved in `outputs/test`.

### Phase 3: Evaluation (Performance Metrics)
Generate final performance metrics (Sharpe Ratio, Max Drawdown, Alpha, etc.).
```bash
python run.py eval
```
*   **Output**: Results are stored in the `results` and `metrics` folders. Check `metrics/summary.json` for the final scorecard.

---

## 🛠️ Configuration Details

You can modify the trading strategy, symbols, and model parameters in:
`configs/main.json`

Key settings:
- `chat_model`: The model to use (default: `meta-llama/Llama-3.3-70B-Instruct`).
- `chat_model_inference_engine`: Set to `local` for offline inference.
- `trading_symbols`: List of assets to trade (e.g., `["BTC", "TSLA"]`).

---

## 📊 Troubleshooting

- **Out of Memory (OOM)**: If your GPU runs out of memory, try using a smaller model (e.g., `Llama-3.1-8B-Instruct`) in `configs/main.json`.
- **Gated Model Error**: Ensure you have been approved by Meta on Hugging Face.
- **Login Issues**: If you use `huggingface-cli login`, ensure your environment can access the stored token (usually automatic in `transformers`).

---
*Note: This agent is for research and evaluation purposes. Use responsibly.*