import os
import gradio as gr
import argparse
from local_notebooklm.processor import generate_audio


def process_audio(pdf_file, format_type, length, style, language, num_speakers, custom_preferences, output_dir):
    if pdf_file is None:
        return "Please upload a PDF file first.", None
    
    if not output_dir:
        output_dir = "./local_notebooklm/web_ui/output"
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    except Exception as e:
        return f"Failed to create output directory: {str(e)}", None
    
    try:
        if hasattr(pdf_file, 'name'):
            pdf_path = pdf_file.name
        else:
            pdf_path = pdf_file
        
        print(f"Processing with output_dir: {output_dir}")
        
        audio_path = generate_audio(
            pdf_path=pdf_path,
            output_dir=output_dir,
            language=language,
            format_type=format_type,
            style=style,
            length=length,
            num_speakers=num_speakers,
            custom_preferences=custom_preferences if custom_preferences else None
        )
        
        if audio_path and os.path.exists(audio_path):
            return "Audio Generated Successfully!", audio_path
        else:
            return "Failed to generate audio.", None
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"An error occurred: {str(e)}\n\nDetails:\n{error_details}", None

def create_gradio_ui():
    format_options = ["podcast"]
    length_options = ["medium"]
    style_options = ["conversational"]
    
    with gr.Blocks(title="Local-NotebookLM") as app:
        gr.Markdown("# 🎙️ Local-NotebookLM: PDF to Audio Converter")
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_file = gr.File(label="Upload PDF", file_types=[".pdf"])
                format_type = gr.Dropdown(choices=format_options, label="Select Format", value=format_options[0])
                length = gr.Dropdown(choices=length_options, label="Select Length", value=length_options[0])
                style = gr.Dropdown(choices=style_options, label="Select Style", value=style_options[0])
                language = gr.Dropdown(
                    choices=["english", "german", "french", "spanish", "italian", "portuguese"],
                    label="Select Language",
                    value="english"
                )
                num_speakers = gr.Number(label="Number of Speakers", value=1, precision=0)
                custom_preferences = gr.Textbox(
                    label="Custom Preferences (Optional)",
                    placeholder="Focus on key points, provide examples, etc."
                )
                output_dir = gr.Textbox(
                    label="Output Directory", 
                    value="./local_notebooklm/web_ui/output",
                    placeholder="Enter the path where output files will be saved"
                )
                generate_button = gr.Button("Generate Audio")
            
            with gr.Column(scale=2):
                result_message = gr.Textbox(label="Status")
                audio_output = gr.Audio(label="Generated Audio", type="filepath")
        
        gr.Markdown("---")
        gr.Markdown("Local-NotebookLM by Gökdeniz Gülmez")
        gr.Markdown("[GitHub Repository](https://github.com/Goekdeniz-Guelmez/Local-NotebookLM)")
        
        generate_button.click(
            fn=process_audio,
            inputs=[pdf_file, format_type, length, style, language, num_speakers, custom_preferences, output_dir],
            outputs=[result_message, audio_output]
        )

    return app

def run_gradio_ui(share=False, port=None):
    app = create_gradio_ui()
    app.launch(share=share, server_port=port)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Local-NotebookLM web UI")
    parser.add_argument("--share", action="store_true", help="Create a shareable link")
    parser.add_argument("--port", type=int, default=None, help="Port to run the interface on")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    run_gradio_ui(share=args.share, port=args.port)

if __name__ == "__main__" or __name__ == "local_notebooklm.web_ui":
    main()