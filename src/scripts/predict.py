import argparse
import os
import pathlib
import google.generativeai as genai

# --- Constants ---
DEFAULT_DATA_DIR = "src/local_data/data"
API_KEY_FILENAME = ".api-gemini"
MODEL_NAME = "models/gemini-2.5-pro" # Using the specified model

# --- Helper Functions ---

def load_api_key(api_key_path: pathlib.Path) -> str | None:
    """Loads the Gemini API key from the specified path."""
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

def generate_gemini_response(api_key: str, prompt_text: str, examples: list[tuple[str, str]]) -> str | None:
    """Generates a response using the Gemini API with few-shot prompting."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(MODEL_NAME)

        # Construct the few-shot prompt
        prompt_parts = [
            "You are an anime chatbot. Given an input dialogue line, generate a relevant response in the style of anime subtitles.",
            "Follow the format of the examples.",
            "\n---\n"
        ]
        for inp, outp in examples:
            prompt_parts.append(f"Input: {inp}")
            prompt_parts.append(f"Output: {outp}")
            prompt_parts.append("---") # Separator between examples

        # Add the actual user input
        prompt_parts.append(f"Input: {prompt_text}")
        prompt_parts.append("Output:") # Ask the model to complete

        full_prompt = "\n".join(prompt_parts)

        # print(f"\n--- Sending Prompt ---\n{full_prompt}\n----------------------\n") # Uncomment for debugging

        response = model.generate_content(full_prompt)

        # print(f"\n--- Received Response ---\n{response}\n-------------------------\n") # Uncomment for debugging

        # Extract text, handling potential blocks or errors
        if response.parts:
             return response.text.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             print(f"Warning: Prompt blocked. Reason: {response.prompt_feedback.block_reason}")
             return f"[Blocked: {response.prompt_feedback.block_reason}]"
        else:
             # Attempt to access text even if parts might be empty (covers edge cases)
             try:
                 return response.text.strip()
             except ValueError: # Handle cases where response.text might raise error
                 print("Warning: Could not extract text from Gemini response.")
                 print(f"Full Response Object: {response}")
                 return "[Error: Could not parse response]"


    except Exception as e:
        print(f"An error occurred calling the Gemini API: {e}")
        # You might want to inspect the specific exception type for more robust handling
        # e.g., handle google.api_core.exceptions.PermissionDenied separately
        return None

# --- Main Prediction Logic ---

def predict(input_file=None, output_file=None, data_dir=DEFAULT_DATA_DIR):
    """
    Loads API key and examples, then generates responses using Gemini API.
    Supports interactive mode or batch processing from a file.
    """
    # --- Load API Key ---
    api_key_path = pathlib.Path.home() / API_KEY_FILENAME
    api_key = load_api_key(api_key_path)
    if not api_key:
        return # Exit if API key is missing

    # --- Load Examples ---
    input_example_path = os.path.join(data_dir, 'input.txt')
    output_example_path = os.path.join(data_dir, 'output.txt')
    examples = load_examples(input_example_path, output_example_path)
    if not examples:
        print("No examples loaded, proceeding without few-shot examples in prompt.")
        # Continue, but the prompt quality might be lower

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
                        response = generate_gemini_response(api_key, prompt, examples)
                        if response:
                            fout.write(response + '\n')
                        else:
                            fout.write("[Error generating response]\n") # Indicate failure
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
            except EOFError: # Handle Ctrl+D or similar EOF signals
                print("\nExiting.")
                break
            if prompt.lower() == 'quit':
                break
            if prompt:
                response = generate_gemini_response(api_key, prompt, examples)
                if response:
                    print(f"Response: {response}")
                else:
                    print("Failed to get response from API.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anime dialogue responses using the Gemini API.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to a file containing input prompts (one per line) for batch mode.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the generated responses in batch mode (defaults to output_responses.txt).")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Directory containing input.txt and output.txt for examples.")

    args = parser.parse_args()
    predict(
        input_file=args.input_file,
        output_file=args.output_file,
        data_dir=args.data_dir,
    )