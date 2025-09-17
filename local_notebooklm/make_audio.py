import argparse, sys

from .processor import generate_audio

# TODO add logging


# Usage
#     generate_audio(
#         pdf_path="/Users/Goekdeniz.Guelmez@computacenter.com/Downloads/why-language-models-hallucinate.pdf",
#         output_dir="/Users/Goekdeniz.Guelmez@computacenter.com/Library/CloudStorage/OneDrive-COMPUTACENTER/Desktop/Local-NotebookLM/output_minimal",
#         llm_model="qwen3:30b-a3b-instruct-2507-q4_K_M",
#         tts_model="kokoro",
#         num_speakers=2,
#         format_type="podcast",
#         style="technical",
#         length="short",
#         custom_preferences="This is a podcast for Computacenter internal, so please welcome the viewer and say that this is for Computacenter internal only."
#     )


def main():
    parser = argparse.ArgumentParser(description="Generate a podcast from a PDF document")
    
    # Required arguments
    parser.add_argument("--pdf", type=str, required=True, help="Path to the PDF file")
    # TODO add the other args
    
    # Optional arguments
    
    args = parser.parse_args()
    
    generate_audio(
        # TODO call with the other args
    )


if __name__ == "__main__":
    sys.exit(main())

if __name__ == "__main__":
    exit(main())