# Anime Subtitle Chatbot with Google Gemini

A chatbot that uses the Google Gemini API (`gemini-2.5-pro`) to generate contextual responses to anime dialogue inputs, using few-shot prompting based on the [Anime Subtitles Dataset](https://www.kaggle.com/jef1056/anime-subtitles).

This project was migrated from an older `trax` implementation and subsequently refactored from a local Hugging Face `distilgpt2` model to use the Gemini API.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/waifuai/anime-subtitle-chatbot-trax # Or your repo URL
    cd anime-subtitle-chatbot-trax
    ```
2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows

    # Or using uv (if installed)
    python -m uv venv .venv
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows
    ```
3.  **Install dependencies:**
    ```bash
    # Using uv (recommended if using uv venv)
    python -m uv pip install -e ./src/anime_chatbot[test]

    # Or using pip
    pip install -e ./src/anime_chatbot[test]
    ```
    This will install the `google-generativeai` library and `pytest` for testing.

4.  **Set up API Key:**
    *   Obtain an API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Create a file named `.api-gemini` in your **home directory** (e.g., `~/.api-gemini` on Linux/macOS or `C:\Users\YourUsername\.api-gemini` on Windows).
    *   Paste your API key into this file on a single line and save it. The script will read the key from this location.

5.  **Prepare Example Data:** (Optional but Recommended for better results)
    *   Place paired input/response text files named `input.txt` (one input phrase per line) and `output.txt` (corresponding response phrases) in the `src/local_data/data/` directory. These will be used for few-shot prompting. The repository includes example files.

## Prediction

Generate responses using the Gemini API via the prediction script:

**Interactive Mode:**
```bash
# Ensure your virtual environment is active
python src/scripts/predict.py
```
Enter input phrases at the prompt. Type `quit` or press `Ctrl+D` to exit.

**Batch Mode:**
```bash
python src/scripts/predict.py --input_file path/to/your_prompts.txt --output_file path/to/save_responses.txt
```
*   `--input_file`: Path to a file with one input prompt per line.
*   `--output_file`: (Optional) Path to save the generated responses. Defaults to `output_responses.txt` in the current directory.
*   `--data_dir`: (Optional) Path to the directory containing `input.txt` and `output.txt` for few-shot examples. Defaults to `src/local_data/data/`. See `python src/scripts/predict.py --help`.

## Project Structure

```
.
├── src/
│   ├── anime_chatbot/          # Core module definition (mainly setup.py)
│   │   ├── __init__.py
│   │   └── setup.py
│   ├── local_data/             # Data storage (gitignored by default)
│   │   └── data/               # Example input/output files for few-shot prompting
│   ├── scripts/                # Python script for prediction
│   │   └── predict.py
│   └── tests/                  # Pytest tests
│       └── test_chatbot.py
├── .gitignore
├── LICENSE
├── pytest.ini
└── README.md                   # This file
```

## Model Architecture

This project uses the Google Gemini API (`gemini-2.5-pro` model) to generate dialogue responses. It employs few-shot prompting, providing the API with examples from `input.txt` and `output.txt` to guide the generation style and context. No local model training or fine-tuning is performed.