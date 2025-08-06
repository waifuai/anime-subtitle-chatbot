import argparse
import os
import pathlib

from anime_chatbot.provider_selector import (
    ProviderConfig,
    resolve_provider,
    generate_response,
)

# --- Constants ---
DEFAULT_DATA_DIR = "src/local_data/data"

# --- Helper Functions ---

def load_api_key(api_key_path: pathlib.Path) -> str | None:
    """Legacy helper kept for backward compatibility in tests."""
    try:
        with open(api_key_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: API key file not found at {api_key_path}")
        print("Please create the file and add your Gemini API key.")
        return None
    except Exception as e:
        print(f"An error occurred reading the API key file: {e}")
        return None

def load_examples(input_path: str, output_path: str) -> list[tuple[str, str]]:
    """Loads dialogue examples from input and output files."""
    examples = []
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'r', encoding='utf-8') as fout:
            for inp_line, out_line in zip(fin, fout):
                inp = inp_line.strip()
                out = out_line.strip()
                if inp and out:
                    examples.append((inp, out))
    except FileNotFoundError:
        print(f"Error: Example file not found at {input_path} or {output_path}")
        print("Please ensure 'input.txt' and 'output.txt' exist.")
        return [] # Return empty list on error
    except Exception as e:
        print(f"An error occurred loading examples: {e}")
        return [] # Return empty list on error
    return examples

def _load_examples(input_path: str, output_path: str) -> list[tuple[str, str]]:
    """Loads dialogue examples from input and output files."""
    examples: list[tuple[str, str]] = []
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'r', encoding='utf-8') as fout:
            for inp_line, out_line in zip(fin, fout):
                inp = inp_line.strip()
                out = out_line.strip()
                if inp and out:
                    examples.append((inp, out))
    except FileNotFoundError:
        print(f"Error: Example file not found at {input_path} or {output_path}")
        print("Please ensure 'input.txt' and 'output.txt' exist.")
        return []
    except Exception as e:
        print(f"An error occurred loading examples: {e}")
        return []
    return examples

# --- Main Prediction Logic ---

def predict(input_file=None, output_file=None, data_dir=DEFAULT_DATA_DIR, provider: str | None = None, model: str | None = None):
    """
    Provider-agnostic prediction. Default provider is OpenRouter.
    Supports interactive mode or batch processing from a file.
    """
    # --- Load Examples ---
    input_example_path = os.path.join(data_dir, 'input.txt')
    output_example_path = os.path.join(data_dir, 'output.txt')
    examples = _load_examples(input_example_path, output_example_path)
    if not examples:
        print("No examples loaded, proceeding without few-shot examples in prompt.")

    cfg = ProviderConfig(
        provider=provider or resolve_provider(None),
        model=model,
        timeout=60,
        max_retries=2,
        retry_backoff_seconds=1.0,
    )

    # --- Mode Selection ---
    if input_file:
        # Batch mode
        output_filename = output_file or "output_responses.txt"
        print(f"Batch mode: Reading from {input_file}, writing to {output_filename}")
        try:
            with open(input_file, 'r', encoding='utf-8') as fin, \
                 open(output_filename, 'w', encoding='utf-8') as fout:
                count = 0
                for line in fin:
                    prompt = line.strip()
                    if prompt:
                        response = generate_response(prompt, examples, cfg)
                        if response:
                            fout.write(response + '\n')
                        else:
                            fout.write("[Error generating response]\n")
                        count += 1
                        if count % 10 == 0:
                            print(f"Processed {count} lines...")
                print(f"Finished processing {count} lines. Output saved to {output_filename}")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file}")
        except Exception as e:
            print(f"An error occurred during batch processing: {e}")
    else:
        # Interactive mode
        print("Interactive mode. Enter 'quit' to exit.")
        while True:
            try:
                prompt = input("Input: ")
            except EOFError:
                print("\nExiting.")
                break
            if prompt.lower() == 'quit':
                break
            if prompt:
                response = generate_response(prompt, examples, cfg)
                if response:
                    print(f"Response: {response}")
                else:
                    print("Failed to get response from API.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anime dialogue responses using configurable providers.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to a file containing input prompts (one per line) for batch mode.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the generated responses in batch mode (defaults to output_responses.txt).")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing input.txt and output.txt for examples.")
    parser.add_argument("--provider", type=str, default=None, choices=["openrouter", "gemini"], help="Provider to use. Defaults to openrouter.")
    parser.add_argument("--model", type=str, default=None, help="Model override. Otherwise reads ~/.model-openrouter or ~/.model-gemini by provider, with sensible defaults.")

    args = parser.parse_args()
    predict(
        input_file=args.input_file,
        output_file=args.output_file,
        data_dir=args.data_dir,
        provider=args.provider,
        model=args.model,
    )