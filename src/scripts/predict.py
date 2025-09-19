"""
Main prediction script for the Anime Subtitle Chatbot.

This module provides the command-line interface and core prediction functionality
for generating anime dialogue responses using configurable AI providers. It supports
both interactive mode for real-time conversations and batch mode for processing
multiple prompts from files.
"""

import argparse
import logging
import os
import pathlib
from typing import Optional

from anime_chatbot.provider_selector import (
    ProviderConfig,
    resolve_provider,
    generate_response,
)

logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_DATA_DIR = "src/local_data/data"

# --- Helper Functions ---

def load_examples(input_path: str, output_path: str) -> list[tuple[str, str]]:
    """
    Loads dialogue examples from input and output files.

    Args:
        input_path: Path to the input examples file.
        output_path: Path to the output examples file.

    Returns:
        List of (input, output) tuples, empty list on error.
    """
    examples: list[tuple[str, str]] = []
    try:
        with open(input_path, 'r', encoding='utf-8') as fin, \
             open(output_path, 'r', encoding='utf-8') as fout:
            for inp_line, out_line in zip(fin, fout):
                inp = inp_line.strip()
                out = out_line.strip()
                if inp and out:  # Skip empty lines
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

def predict(input_file: Optional[str] = None,
           output_file: Optional[str] = None,
           data_dir: str = DEFAULT_DATA_DIR,
           provider: Optional[str] = None,
           model: Optional[str] = None) -> None:
    """
    Provider-agnostic prediction with robust batch and interactive modes.

    Args:
        input_file: Path to file with prompts (one per line) for batch mode.
        output_file: Path to save responses in batch mode.
        data_dir: Directory containing input.txt and output.txt for examples.
        provider: Provider to use ('openrouter' or 'gemini').
        model: Model override for the selected provider.
    """
    logger.info("Starting prediction process")

    # Validate data directory
    if not os.path.isdir(data_dir):
        error_msg = f"Data directory not found: {data_dir}"
        logger.error(error_msg)
        print(error_msg)
        return

    # Load examples
    input_example_path = os.path.join(data_dir, 'input.txt')
    output_example_path = os.path.join(data_dir, 'output.txt')
    examples = load_examples(input_example_path, output_example_path)

    if examples:
        logger.info(f"Loaded {len(examples)} examples from {data_dir}")
    else:
        logger.warning("No examples loaded, proceeding without few-shot examples")

    # Configure provider
    resolved_provider = provider or resolve_provider(None)
    cfg = ProviderConfig(
        provider=resolved_provider,
        model=model,
        timeout=60,
        max_retries=2,
        retry_backoff_seconds=1.0,
    )
    logger.info(f"Using provider: {resolved_provider}, model: {model or 'default'}")

    # Mode selection
    if input_file:
        _run_batch_mode(input_file, output_file, examples, cfg)
    else:
        _run_interactive_mode(examples, cfg)

    logger.info("Prediction process completed")


def _run_batch_mode(input_file: str,
                   output_file: Optional[str],
                   examples: list[tuple[str, str]],
                   cfg: ProviderConfig) -> None:
    """Handle batch processing mode with enhanced validation."""
    logger.info(f"Starting batch mode with input: {input_file}")

    # Validate input file
    if not os.path.isfile(input_file):
        error_msg = f"Input file not found: {input_file}"
        logger.error(error_msg)
        print(error_msg)
        return

    # Check if input file is empty
    if os.path.getsize(input_file) == 0:
        error_msg = f"Input file is empty: {input_file}"
        logger.error(error_msg)
        print(error_msg)
        return

    output_filename = output_file or "output_responses.txt"
    logger.info(f"Batch mode: Writing responses to {output_filename}")

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_filename, 'w', encoding='utf-8') as fout:

            count = 0
            success_count = 0
            error_count = 0

            for line_num, line in enumerate(fin, 1):
                prompt = line.strip()
                if not prompt:  # Skip empty lines
                    continue

                if len(prompt) > 1000:  # Reasonable limit
                    logger.warning(f"Line {line_num}: Prompt too long ({len(prompt)} chars), truncating")
                    prompt = prompt[:1000]

                logger.debug(f"Processing line {line_num}: {prompt[:50]}...")
                response = generate_response(prompt, examples, cfg)

                if response:
                    fout.write(response + '\n')
                    fout.flush()  # Ensure immediate write
                    success_count += 1
                    logger.debug(f"Line {line_num}: Response generated successfully")
                else:
                    error_msg = f"[Error generating response for: {prompt[:50]}...]"
                    fout.write(error_msg + '\n')
                    error_count += 1
                    logger.warning(f"Line {line_num}: Failed to generate response")

                count += 1
                if count % 10 == 0:
                    progress_msg = f"Processed {count} lines... ({success_count} success, {error_count} errors)"
                    print(progress_msg)
                    logger.info(progress_msg)

            summary_msg = f"Batch processing complete: {count} total, {success_count} success, {error_count} errors"
            print(f"\n{summary_msg}")
            logger.info(f"{summary_msg}. Output saved to {output_filename}")

    except FileNotFoundError:
        error_msg = f"Error: Input file not found at {input_file}"
        logger.error(error_msg)
        print(error_msg)
    except PermissionError as e:
        error_msg = f"Permission denied accessing files: {e}"
        logger.error(error_msg)
        print(error_msg)
    except UnicodeDecodeError as e:
        error_msg = f"Encoding error reading input file: {e}"
        logger.error(error_msg)
        print(error_msg)
    except Exception as e:
        error_msg = f"An unexpected error occurred during batch processing: {e}"
        logger.error(error_msg)
        print(error_msg)


def _run_interactive_mode(examples: list[tuple[str, str]], cfg: ProviderConfig) -> None:
    """Handle interactive mode with better user experience."""
    logger.info("Starting interactive mode")

    print("Interactive mode. Type 'quit' or 'exit' to end the session.")
    print("Type 'help' for available commands.\n")

    while True:
        try:
            prompt = input("Input: ").strip()
        except EOFError:
            print("\nExiting due to EOF (Ctrl+D).")
            logger.info("Interactive mode ended by EOF")
            break
        except KeyboardInterrupt:
            print("\nExiting due to user interrupt (Ctrl+C).")
            logger.info("Interactive mode ended by user interrupt")
            break

        # Handle special commands
        if prompt.lower() in ('quit', 'exit'):
            print("Goodbye!")
            logger.info("Interactive mode ended by user command")
            break
        elif prompt.lower() == 'help':
            _show_interactive_help()
            continue
        elif prompt.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue
        elif not prompt:
            continue  # Skip empty input

        # Validate input length
        if len(prompt) > 1000:
            print("Warning: Input too long (max 1000 chars). Truncating...")
            prompt = prompt[:1000]

        # Generate response
        logger.info(f"Processing interactive input: {prompt[:50]}...")
        response = generate_response(prompt, examples, cfg)

        if response:
            print(f"Response: {response}")
            logger.info("Interactive response generated successfully")
        else:
            error_msg = "Failed to get response from API. Please try again."
            print(error_msg)
            logger.warning("Interactive response generation failed")


def _show_interactive_help() -> None:
    """Display help information for interactive mode."""
    help_text = """
Available commands:
  quit/exit  - End the session
  help       - Show this help message
  clear      - Clear the screen

Simply type any text to generate a response!
"""
    print(help_text)

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