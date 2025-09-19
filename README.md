# Anime Subtitle Chatbot with Multiple AI Providers

A flexible chatbot that generates contextual responses to anime dialogue inputs using few-shot prompting based on the [Anime Subtitles Dataset](https://www.kaggle.com/jef1056/anime-subtitles). The project supports multiple AI providers through a modular architecture.

This project was migrated from an older `trax` implementation and subsequently refactored from a local Hugging Face `distilgpt2` model to use external AI APIs with a provider-based architecture.

## What's New: Provider-Based Architecture

The project now uses a modular provider system supporting multiple AI services:
- **Google Gemini**: Uses the official Google GenAI SDK with centralized client management
- **OpenRouter**: Provides access to multiple models through a unified API
- **Extensible**: Easy to add new providers through the provider interface

### Supported Providers

- **Gemini** (`gemini`): Google Gemini models with robust error handling and content filtering
- **OpenRouter** (`openrouter`): Access to various models including DeepSeek with retry logic

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

4.  Set up API Keys (choose one provider or both):
    - **For Gemini**:
      - Preferred: set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your environment.
      - Fallback: create a single-line file at `~/.api-gemini` (Windows: `C:\Users\YourUsername\.api-gemini`) containing your key.
      - Get a key from: https://aistudio.google.com/app/apikey
    - **For OpenRouter**:
      - Preferred: set `OPENROUTER_API_KEY` in your environment.
      - Fallback: create a single-line file at `~/.api-openrouter` (Windows: `C:\Users\YourUsername\.api-openrouter`) containing your key.
      - Get a key from: https://openrouter.ai/
    - **Provider Selection**:
      - Default: OpenRouter (more cost-effective)
      - Set `CHATBOT_PROVIDER=gemini` to use Gemini
      - Use `--provider` CLI flag to override

5.  Prepare Example Data (optional, improves style consistency):
    - Place `input.txt` and `output.txt` under `src/local_data/data/`. Example files are included.

## Usage

### Interactive Mode
```bash
.venv/Scripts/python.exe -m python src/scripts/predict.py
```
Type `quit` or press `Ctrl+D` to exit.

### Batch Mode
```bash
.venv/Scripts/python.exe -m python src/scripts/predict.py --input_file path/to/prompts.txt --output_file path/to/responses.txt
```

### Provider Selection
Use Gemini:
```bash
.venv/Scripts/python.exe -m python src/scripts/predict.py --provider gemini
```

Use OpenRouter:
```bash
.venv/Scripts/python.exe -m python src/scripts/predict.py --provider openrouter
```

Custom model:
```bash
.venv/Scripts/python.exe -m python src/scripts/predict.py --provider gemini --model gemini-1.5-pro
```

### Command Line Options
- `--input_file`: text file with one prompt per line for batch processing
- `--output_file`: optional output path (default: `./output_responses.txt`)
- `--data_dir`: optional few-shot data directory (default: `src/local_data/data/`)
- `--provider`: AI provider to use (`openrouter` or `gemini`, default: `openrouter`)
- `--model`: specific model to use (provider-dependent, overrides defaults)

## Project Structure

```
.
├── src/
│   ├── anime_chatbot/           # Core package
│   │   ├── __init__.py         # Package exports and convenience imports
│   │   ├── setup.py            # Package setup and metadata
│   │   ├── provider_selector.py # Provider selection and configuration logic
│   │   ├── providers/          # AI provider implementations
│   │   │   ├── gemini.py       # Google Gemini provider
│   │   │   └── openrouter.py   # OpenRouter provider
│   │   └── README.md           # Package documentation
│   ├── local_data/             # Example data for few-shot prompting
│   │   └── data/
│   │       ├── input.txt       # Example dialogue inputs
│   │       ├── output.txt      # Example dialogue outputs
│   │       └── train.txt       # Training data (legacy)
│   ├── scripts/               # Executable scripts
│   │   └── predict.py         # Main prediction script with CLI
│   └── tests/                # Test suite
│       └── test_chatbot.py   # Comprehensive test coverage
├── .gitignore                 # Git ignore patterns
├── LICENSE                    # MIT-0 License
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Models and Providers

The project supports multiple AI models through different providers:

### Default Models
- **OpenRouter (default)**: `deepseek/deepseek-chat-v3-0324:free`
- **Gemini**: `models/gemini-2.5-pro`

### Custom Model Selection
- Create `~/.model-openrouter` file with custom OpenRouter model name
- Create `~/.model-gemini` file with custom Gemini model name
- Use `--model` CLI parameter to override
- Set environment variables `OPENROUTER_API_KEY` or `GEMINI_API_KEY`

### Few-Shot Prompting
- Uses examples from `src/local_data/data/input.txt` and `output.txt`
- No fine-tuning or local training is performed
- Examples improve style consistency and response quality

## Development

### Running Tests
```bash
# Install with test dependencies
.venv/Scripts/python.exe -m uv pip install -e ./src/anime_chatbot[test]

# Run all tests
.venv/Scripts/python.exe -m pytest

# Run with coverage
.venv/Scripts/python.exe -m pytest --cov=src/anime_chatbot --cov-report=html
```

### Adding New Providers
1. Create a new provider class in `src/anime_chatbot/providers/`
2. Implement the required interface (see existing providers)
3. Add the provider to `provider_selector.py`
4. Update tests and documentation

### Project Architecture
- **Provider Pattern**: Clean separation between different AI services
- **Configuration Management**: Environment variables and file-based config
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Testing**: Mock-based testing for reliable CI/CD
- **CLI Interface**: Simple command-line interface with multiple options