# Anime Subtitle Chatbot with Google Gemini

A chatbot that uses the Google Gemini API (`gemini-2.5-pro`) to generate contextual responses to anime dialogue inputs, using few-shot prompting based on the [Anime Subtitles Dataset](https://www.kaggle.com/jef1056/anime-subtitles).

This project was migrated from an older `trax` implementation and subsequently refactored from a local Hugging Face `distilgpt2` model to use the Gemini API.

## What's New: Google GenAI SDK

The project now uses the official Google GenAI SDK with the centralized client:
- Import: `from google import genai`
- Client: `client = genai.Client(api_key=...)` (env var preferred; key file fallback)
- Generation: `client.models.generate_content(model="gemini-2.5-pro", contents=...)`

Benefits: unified auth/config, simpler extension to chats/files/streaming. The previous `google.generativeai` is no longer used.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/waifuai/anime-subtitle-chatbot-trax
    cd anime-subtitle-chatbot-trax
    ```

2.  Create the uv virtual environment and ensure tooling (Windows venv shim path shown):
    ```bash
    python -m uv venv .venv
    .venv/Scripts/python.exe -m ensurepip
    .venv/Scripts/python.exe -m pip install uv
    ```

3.  Install dependencies:
    ```bash
    .venv/Scripts/python.exe -m uv pip install -r requirements.txt
    .venv/Scripts/python.exe -m uv pip install -e ./src/anime_chatbot[test]
    ```

4.  Set up API Key:
    - Preferred: set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your environment.
    - Fallback: create a single-line file at `~/.api-gemini` (Windows: `C:\Users\YourUsername\.api-gemini`) containing your key.
    - Get a key from: https://aistudio.google.com/app/apikey

5.  Prepare Example Data (optional, improves style consistency):
    - Place `input.txt` and `output.txt` under `src/local_data/data/`. Example files are included.

## Prediction

Interactive mode:
```bash
.venv/Scripts/python.exe -m python src/scripts/predict.py
```
Type `quit` or press `Ctrl+D` to exit.

Batch mode:
```bash
.venv/Scripts/python.exe -m python src/scripts/predict.py --input_file path/to/prompts.txt --output_file path/to/responses.txt
```
- `--input_file`: text file with one prompt per line
- `--output_file`: optional output path (default: `./output_responses.txt`)
- `--data_dir`: optional few-shot data directory (default: `src/local_data/data/`)

## Project Structure

```
.
├── src/
│   ├── anime_chatbot/
│   │   ├── __init__.py
│   │   └── setup.py
│   ├── local_data/
│   │   └── data/
│   ├── scripts/
│   │   └── predict.py
│   └── tests/
│       └── test_chatbot.py
├── .gitignore
├── LICENSE
├── pytest.ini
├── requirements.txt
└── README.md
```

## Model

Standardized on `gemini-2.5-pro`. Few-shot prompting uses `src/local_data/data/input.txt` and `output.txt`. No fine-tuning or local training is performed.