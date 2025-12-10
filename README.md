# Local-NotebookLM

![logo](logo.jpeg)

A local AI-powered tool that converts PDF documents into engaging audio's such as podcasts, using local LLMs and TTS models.

## Features

- PDF text extraction and processing
- Customizable podcast generation with different styles and lengths
- Support for various LLM providers (OpenAI, Groq, LMStudio, Ollama, Azure)
- Text-to-Speech conversion with voice selection
- Fully configurable pipeline
- Preference-based content focus
- Programmatic API for integration in other projects
- FastAPI server for web-based access
- Example podcast included for demonstration

#### Here is a quick example, can you guess what paper they're talking about?

<audio controls>
  <source src="./examples/podcast.wav" type="audio/mpeg">
  Your browser does not support the audio element. You can manualy download the file here './examples/podcast.wav'.
</audio>

## Prerequisites

- Python 3.12+
- Local LLM server (optional, for local inference)
- Local TTS server (optional, for local audio generation)
- At least 8GB RAM (16GB+ recommended for local models)
- 10GB+ free disk space

## Installation

### From PyPI

```bash
pip install local-notebooklm
```

### From source

1. Clone the repository:

```bash
git clone https://github.com/Goekdeniz-Guelmez/Local-NotebookLM.git
cd Local-NotebookLM
```

2. Create and activate a virtual environment (conda works too):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Running with Docker Compose

You can also run both the Gradio Web UI and FastAPI server using Docker Compose.

### Prerequisites

- Docker and Docker Compose installed on your system

### Steps

1. Open a terminal and navigate to the `docker/` folder inside the project:

```bash
cd docker
```

2. Build and start the containers:

```bash
docker-compose up --build
```

This command will:

- Start the Gradio Web UI at [http://localhost:7860](http://localhost:7860)
- Start the FastAPI server at [http://localhost:8000](http://localhost:8000)

You can access the web interface or use the API endpoints after running the command.

To stop the services, press `CTRL+C` and then run:

```bash
docker-compose down
```
## Optional pre requisites
### Local TTS server
- Follow one installation type (docker, docker-compose, uv) at https://github.com/remsky/Kokoro-FastAPI
- Test in your browser that http://localhost:8880/v1 return the json: {"detail":"Not Found"}
  
## Example Output

The repository includes an example podcast in `examples/podcast.wav` to demonstrate the quality and format of the output. The models used are: gpt4o and Mini with tts-hs on Azure. You can listen to this example to get a sense of what Local-NotebookLM can produce before running it on your own PDFs.

## Configuration

You can use the default configuration or create a custom JSON config file with the following structure:

```json
{
    "Co-Host-Speaker-1-Voice": "af_sky+af_bella",
    "Co-Host-Speaker-2-Voice": "af_echo",
    "Co-Host-Speaker-3-Voice": "af_nova",
    "Co-Host-Speaker-4-Voice": "af_shimmer",
    "Host-Speaker-Voice": "af_alloy",

    "Small-Text-Model": {
        "provider": {
            "name": "groq",
            "key": "your-api-key"
        },
        "model": "llama-3.2-90b-vision-preview"
    },

    "Big-Text-Model": {
        "provider": {
            "name": "groq",
            "key": "your-api-key"
        },
        "model": "llama-3.2-90b-vision-preview"
    },

    "Text-To-Speech-Model": {
        "provider": {
            "name": "custom",
            "endpoint": "http://localhost:8880/v1",
            "key": "not-needed"
        },
        "model": "kokoro",
        "audio_format": "wav"
    },

    "Step1": {
        "system": "",
        "max_tokens": 1028,
        "temperature": 0.7,
        "chunk_size": 1000,
        "max_chars": 100000
    },

    "Step2": {
        "system": "",
        "max_tokens": 8126,
        "temperature": 1,
        "chunk_token_limit": 2000,
        "overlap_percent": 10
    },

    "Step3": {
        "system": "",
        "max_tokens": 8126,
        "temperature": 1,
        "chunk_token_limit": 2000,
        "overlap_percent": 20
    }
}
```

### Provider Options

The following provider options are supported:

- **OpenAI**: Use OpenAI's API
  ```json
  "provider": {
      "name": "openai",
      "key": "your-openai-api-key"
  }
  ```

- **Groq**: Use Groq's API for faster inference
  ```json
  "provider": {
      "name": "groq",
      "key": "your-groq-api-key"
  }
  ```

- **Azure OpenAI**: Use Azure's OpenAI service
  ```json
  "provider": {
      "name": "azure",
      "key": "your-azure-api-key",
      "endpoint": "your-azure-endpoint",
      "version": "api-version"
  }
  ```

- **LMStudio**: Use a local LMStudio server
  ```json
  "provider": {
      "name": "lmstudio",
      "endpoint": "http://localhost:1234/v1",
      "key": "not-needed"
  }
  ```

- **Ollama**: Use a local Ollama server
  ```json
  "provider": {
      "name": "ollama",
      "endpoint": "http://localhost:11434",
      "key": "not-needed"
  }
  ```

- **Google generative AI**: Use Google's API
  ```json
  "provider": {
      "name": "google",
      "key": "your-google-genai-api-key"
  }
  ```

- **Anthropic**: Use Anthropic's API
  ```json
  "provider": {
      "name": "anthropic",
      "key": "your-anthropic-api-key"
  }
  ```

- **Elevenlabs**: Use Elevenlabs's API
  ```json
  "provider": {
      "name": "elevenlabs",
      "key": "your-elevenlabs-api-key"
  }
  ```

- **Custom**: Use any OpenAI-compatible API
  ```json
  "provider": {
      "name": "custom",
      "endpoint": "your-custom-endpoint",
      "key": "your-api-key-or-not-needed"
  }
  ```

## Usage

### Command Line Interface

Run the script with the following command:

```bash
python -m local_notebooklm.start --pdf PATH_TO_PDF [options]
```

#### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--pdf` | Path to the PDF file (required) | - |
| `--config` | Path to custom config file | Uses base_config |
| `--format` | Output format type (summary, podcast, article, interview, panel-discussion, debate, narration, storytelling, explainer, lecture, tutorial, q-and-a, news-report, executive-brief, meeting, analysis) | podcast |
| `--length` | Content length (short, medium, long, very-long) | medium |
| `--style` | Content style (normal, casual, formal, technical, academic, friendly, gen-z, funny) | normal |
| `--preference` | Additional focus preferences or instructions | None |
| `--language` | Language the audio should be in | english |
| `--output-dir` | Directory to store output files | ./output |

#### Format Types

Local-NotebookLM now supports both single-speaker and two-speaker formats:

**Single-Speaker Formats:**
- summary
- narration
- storytelling
- explainer
- lecture
- tutorial
- news-report
- executive-brief
- analysis

**Two-Speaker Formats:**
- podcast
- interview
- panel-discussion
- debate
- q-and-a
- meeting

**Multi-Speaker Formats:**
- panel-discussion (3, 4, or 5 speakers)
- debate (3, 4, or 5 speakers)

#### Example Commands

Basic usage:
```bash
python -m local_notebooklm.start --pdf documents/research_paper.pdf
```

Customized podcast:
```bash
python -m local_notebooklm.start --pdf documents/research_paper.pdf --format podcast --length long --style casual
```

With custom preferences:
```bash
python -m local_notebooklm.start --pdf documents/research_paper.pdf --preference "Focus on practical applications and real-world examples"
```

Using custom config:
```bash
python -m local_notebooklm.start --pdf documents/research_paper.pdf --config custom_config.json --output-dir ./my_podcast --language german
```

### Programmatic API

You can also use Local-NotebookLM programmatically in your Python code:

```python
from local_notebooklm.processor import podcast_processor

success, result = podcast_processor(
    pdf_path="documents/research_paper.pdf",
    config_path="config.json",
    format_type="interview",
    length="long",
    style="professional",
    preference="Focus on the key technical aspects",
    output_dir="./test_output",
    language="english"
)

if success:
    print(f"Successfully generated podcast: {result}")
else:
    print(f"Failed to generate podcast: {result}")
```

### Gradio Web UI

Local-NotebookLM now includes a user-friendly Gradio web interface that makes it easy to use the tool without command line knowledge:

```bash
python -m local_notebooklm.web_ui
```

By default, the web UI runs locally on http://localhost:7860. You can access it from your browser.

#### Web UI Screenshots

![Web UI Main Screen](examples/Gradio-WebUI.png)
*The main interface of the Local-NotebookLM web UI*

#### Web UI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--share` | Make the UI accessible over the network | False |
| `--port` | Specify a custom port | 7860 |

#### Example Commands

Basic local usage:
```bash
python -m local_notebooklm.web_ui
```

Share with others on your network:
```bash
python -m local_notebooklm.web_ui --share
```

Use a custom port:
```bash
python -m local_notebooklm.web_ui --port 8080
```

The web interface provides all the same options as the command line tool in an intuitive UI, making it easier for non-technical users to generate audio content from PDFs.

### FastAPI Server

Start the FastAPI server to access the functionality via a web API:

```bash
 python -m local_notebooklm.server
```

By default, the server runs on http://localhost:8000. You can access the API documentation at http://localhost:8000/docs.

## Pipeline Steps

### 1. PDF Processing (Step1)
- Extracts text from PDF documents
- Cleans and formats the content
- Removes irrelevant elements like page numbers and headers
- Handles LaTeX math expressions and special characters
- Splits content into manageable chunks for processing

### 2. Transcript Generation (Step2)
- Generates an initial podcast script based on the extracted content
- Applies the specified style (casual, formal, technical, academic)
- Formats content according to the desired length (short, medium, long, very-long)
- Structures content for a conversational format
- Incorporates user-specified format type (summary, podcast, article, interview)

### 3. TTS Optimization (Step3)
- Rewrites content specifically for better text-to-speech performance
- Creates a two-speaker conversation format
- Adds speech markers and natural conversation elements
- Optimizes for natural flow and engagement
- Incorporates user preferences for content focus
- Formats output as a list of speaker-text tuples

### 4. Audio Generation (Step4)
- Converts the optimized text to speech using the specified TTS model
- Applies different voices for each speaker
- Generates individual audio segments for each dialogue part
- Concatenates segments into a final audio file
- Maintains consistent audio quality and sample rate

### Here is a detaled diagram to visualize the architecture of my project.

```mermaid
flowchart TD
    subgraph "Main Controller"
        processor["podcast_processor()"]
    end

    subgraph "AI Services"
        smallAI["Small Text Model Client"]
        bigAI["Big Text Model Client"]
        ttsAI["Text-to-Speech Model Client"]
    end
    
    subgraph "Step 1: PDF Processing"
        s1["step1()"]
        validate["validate_pdf()"]
        extract["extract_text_from_pdf()"]
        chunk1["create_word_bounded_chunks()"]
        process["process_chunk()"]
    end
    
    subgraph "Step 2: Transcript Generation"
        s2["step2()"]
        read2["read_input_file()"]
        gen2["generate_transcript()"]
        chunk2["Chunking with Overlap"]
    end
    
    subgraph "Step 3: TTS Optimization"
        s3["step3()"]
        read3["read_pickle_file()"]
        gen3["generate_rewritten_transcript()"]
        genOverlap["generate_rewritten_transcript_with_overlap()"]
        validate3["validate_transcript_format()"]
    end
    
    subgraph "Step 4: Audio Generation"
        s4["step4()"]
        load4["load_podcast_data()"]
        genAudio["generate_speaker_audio()"]
        concat["concatenate_audio_files()"]
    end

    %% Flow connections
    processor --> s1
    processor --> s2
    processor --> s3
    processor --> s4
    
    processor -.-> smallAI
    processor -.-> bigAI
    processor -.-> ttsAI
    
    %% Step 1 flow
    s1 --> validate
    validate --> extract
    extract --> chunk1
    chunk1 --> process
    process -.-> smallAI
    
    %% Step 2 flow
    s2 --> read2
    read2 --> gen2
    gen2 --> chunk2
    gen2 -.-> bigAI
    
    %% Step 3 flow
    s3 --> read3
    read3 --> gen3
    read3 --> genOverlap
    gen3 --> validate3
    genOverlap --> validate3
    gen3 -.-> bigAI
    genOverlap -.-> bigAI
    
    %% Step 4 flow
    s4 --> load4
    load4 --> genAudio
    genAudio --> concat
    genAudio -.-> ttsAI
    
    %% Data flow
    pdf[("PDF File")] --> s1
    s1 --> |"cleaned_text.txt"| file1[("Cleaned Text")]
    file1 --> s2
    s2 --> |"data.pkl"| file2[("Transcript")]
    file2 --> s3
    s3 --> |"podcast_ready_data.pkl"| file3[("Optimized Transcript")]
    file3 --> s4
    s4 --> |"podcast.wav"| fileAudio[("Final Audio")]

    %% Styling
    classDef controller fill:#f9d5e5,stroke:#333,stroke-width:2px
    classDef ai fill:#eeeeee,stroke:#333,stroke-width:1px
    classDef step fill:#d0e8f2,stroke:#333,stroke-width:1px
    classDef data fill:#fcf6bd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5
    
    class processor controller
    class smallAI,bigAI,ttsAI ai
    class s1,s2,s3,s4,validate,extract,chunk1,process,read2,gen2,chunk2,read3,gen3,genOverlap,validate3,load4,genAudio,concat step
    class pdf,file1,file2,file3,fileAudio data
```

## Multiple Language Support

Local-NotebookLM now supports multiple languages. You can specify the language when using the programmatic API or through the command line.

**Important Note:** When using a non-English language, ensure that both your selected LLM and TTS models support the desired language. Language support varies significantly between different models and providers. For optimal results, verify that your chosen models have strong capabilities in your target language before processing.


## Output Files

The pipeline generates the following files:

- `step1/extracted_text.txt`: Raw text extracted from the PDF
- `step1/clean_extracted_text.txt`: Cleaned and processed text
- `step2/data.pkl`: Initial transcript data
- `step3/podcast_ready_data.pkl`: TTS-optimized conversation data
- `step4/segments/podcast_segment_*.wav`: Individual audio segments
- `step4/podcast.wav`: Final concatenated podcast audio file

## Troubleshooting

### Common Issues

1. **PDF Extraction Fails**
   - Try a different PDF file
   - Check if the PDF is password-protected
   - Ensure the PDF contains extractable text (not just images)

2. **API Connection Errors**
   - Verify your API keys are correct
   - Check your internet connection
   - Ensure the API endpoints are accessible

3. **Out of Memory Errors**
   - Reduce the chunk size in the configuration
   - Use a smaller model
   - Close other memory-intensive applications

4. **Audio Quality Issues**
   - Try different TTS voices
   - Adjust the sample rate in the configuration
   - Check if the TTS server is running correctly

### Getting Help

If you encounter issues not covered here, please:
1. Check the logs for detailed error messages
2. Open an issue on the GitHub repository with details about your problem
3. Include the error message and steps to reproduce the issue

## Requirements

- Python 3.12+
- PyPDF2
- tqdm
- numpy
- soundfile
- requests
- pathlib
- fastapi
- uvicorn

Full requirements are listed in `requirements.txt`.

## Acknowledgments

- This project uses various open-source libraries and models
- Special thanks to the developers of LLaMA, OpenAI, and other AI models that make this possible

---

For more information, visit the [GitHub repository](https://github.com/Goekdeniz-Guelmez/Local-NotebookLM).

Best
Gökdeniz Gülmez

---

![Alt](https://repobeats.axiom.co/api/embed/28af9fd2bc35cdc4974f5766dd60c1fa9323a4a2.svg "Repobeats analytics image")

---

## Citing Local-NotebookLM

The Local-NotebookLM software suite was developed by Gökdeniz Gülmez. If you find Local-NotebookLM useful in your research and wish to cite it, please use the following
BibTex entry:

```text
@software{
  Local-NotebookLM,
  author = {Gökdeniz Gülmez},
  title = {{Local-NotebookLM}: A Local-NotebookLM to convert PDFs into Audio.},
  url = {https://github.com/Goekdeniz-Guelmez/Local-NotebookLM},
  version = {0.1.5},
  year = {2025},
}
```
