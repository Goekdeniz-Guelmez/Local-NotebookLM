import sys
import json
import logging
from pathlib import Path
from typing import List, Tuple, Literal, Optional

import PyPDF2
import numpy as np
import soundfile as sf
from openai import OpenAI


# ==== Colored logging formatter ====
class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

handler = logging.StreamHandler(sys.stderr)
handler.setLevel(logging.DEBUG)
formatter = ColoredFormatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger = logging.getLogger("ai_audio")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False

# ========== PDF EXTRACTION ==========
def extract_text_from_pdf(pdf_path: str, max_chars: int = 100_000) -> str:
    if not Path(pdf_path).exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        logger.debug(f"PDF has {num_pages} pages.")
        text = []
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if total + len(page_text) > max_chars:
                text.append(page_text[: max_chars - total])
                break
            text.append(page_text)
            total += len(page_text)
    return "\n".join(text)

def chunk_text(
    text: str,
    length: Literal["short", "medium", "long"] = "medium"
) -> List[str]:
    """Split text into overlapping chunks with size based on desired output length.

    The `length` parameter controls how aggressively we chunk the PDF text:
    - short: smaller chunks, fewer total chunks
    - medium: balanced chunk size and count
    - long: larger chunks, allow many chunks
    """
    # Chunking presets tuned for typical LLM context windows (characters approximation)
    presets = {
        "short":  {"max_chunk_size": 30_000, "overlap": 300, "max_chunks": 2},   # biggest chunks → short summary
        "medium": {"max_chunk_size": 15_000, "overlap": 300, "max_chunks": 4},   # balanced
        "long":   {"max_chunk_size": 5_000,  "overlap": 200, "max_chunks": 9999999}, # smallest chunks → detailed
    }
    cfg = presets[length]
    logger.debug(f"Chunking config chosen: {cfg} for length='{length}'")

    chunks: List[str] = []
    start = 0
    text_length = len(text)

    while start < text_length and len(chunks) < cfg["max_chunks"]:
        end = min(start + cfg["max_chunk_size"], text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start = end - cfg["overlap"]

    logger.info(
        f"Total chunks created: {len(chunks)} (length={length}, max_chunk_size={cfg['max_chunk_size']}, overlap={cfg['overlap']})"
    )
    return chunks

def generate_transcript_from_pdf(
    pdf_path: str,
    client: OpenAI,
    model: str,
    language: str = "english",
    format_type: Literal[
        "podcast", "narration", "interview", "panel-discussion", "summary", "article", "lecture",
        "q-and-a", "tutorial", "debate", "meeting", "analysis"
    ] = "podcast",
    style: Literal[
        "normal", "formal", "casual", "enthusiastic", "serious", "humorous", "gen-z", "technical"
    ] = "normal",
    length: Literal["short", "medium", "long"] = "medium",
    num_speakers: Optional[int] = None,
    custom_preferences: Optional[str] = None
) -> List[Tuple[str, str]]:
    logger.info("Extracting large text from PDF...")
    length_max_chars = {"short": 200_000, "medium": 600_000, "long": 1_200_000}
    text = extract_text_from_pdf(pdf_path, max_chars=length_max_chars.get(length, 600_000))
    logger.debug(f"Extracted text length: {len(text)} characters.")
    logger.info("Splitting text into chunks...")
    chunks = chunk_text(text, length=length)
    total_chunks = len(chunks)
    logger.info(f"Starting to process {total_chunks} chunks for transcript generation...")
    all_transcript = []
    for i, chunk in enumerate(chunks):
        is_first = (i == 0)
        is_last = (i == total_chunks - 1)
        logger.info(f"Generating transcript for chunk {i+1}/{total_chunks}, is_first={is_first}, is_last={is_last}")
        # Build context from previous turns
        prev_context = ""
        if all_transcript:
            context_slice = all_transcript[-3:]
            prev_context = "\n".join([f"{s}: {t}" for s, t in context_slice])
        if prev_context:
            combined_chunk = prev_context + "\n\n[Previous context]\n" + chunk
        else:
            combined_chunk = chunk
        # Append special markers for intro/outro control
        if is_first:
            combined_chunk += "\n\n[Include an opening greeting appropriate to the format.]"
        elif is_last:
            combined_chunk += "\n\n[Include a closing goodbye appropriate to the format.]"
        else:
            combined_chunk += "\n\n[Do not include greetings or closings in this chunk.]"
        # Call generate_characters with num_speakers and custom_preferences
        chunk_transcript = generate_transcript(
            combined_chunk, client, model, language, format_type=format_type,
            is_first=is_first, is_last=is_last, style=style, length=length,
            num_speakers=num_speakers, custom_preferences=custom_preferences
        )
        all_transcript.extend(chunk_transcript)
    logger.info("Completed generating transcripts for all chunks.")
    return all_transcript

# ========== LLM TRANSCRIPT ==========
def generate_transcript(
    text: str,
    client: OpenAI,
    model: str,
    language: str = "english",
    format_type: Literal[
        "podcast", "narration", "interview", "panel-discussion", "summary", "article", "lecture",
        "q-and-a", "tutorial", "debate", "meeting", "analysis"
    ] = "podcast",
    is_first: bool = False,
    is_last: bool = False,
    style: Literal[
        "normal", "formal", "casual", "enthusiastic", "serious", "humorous", "gen-z", "technical"
    ] = "normal",
    length: Literal["short", "medium", "long"] = "medium",
    num_speakers: Optional[int] = None,
    custom_preferences: Optional[str] = None
) -> List[Tuple[str, str]]:
    """
    Generate a audio transcript as a list of (speaker, text) tuples.
    """
    import json

    format_guides = {
        "podcast": "A conversational and engaging format with multiple speakers, natural flow, and informal tone.",
        "narration": "A single speaker delivering clear, concise, and informative narration.",
        "interview": "Two speakers with a question-and-answer format, focused and interactive.",
        "panel-discussion": "Multiple speakers engaging in a dynamic and balanced discussion.",
        "summary": "A brief and clear overview of the main points.",
        "article": "Structured, formal, and informative content suitable for reading.",
        "lecture": "Educational and detailed explanation with a formal tone.",
        "q-and-a": "Question and answer style with clear, concise responses.",
        "tutorial": "Step-by-step instructional content with clear guidance.",
        "debate": "Contrasting viewpoints presented by multiple speakers in a formal style.",
        "meeting": "Professional and concise discussion among participants.",
        "analysis": "In-depth examination and interpretation with expert tone."
    }

    style_guides = {
        "normal": "Use a balanced and natural conversational style appropriate for the format.",
        "formal": "Use a professional and polished tone with proper grammar and vocabulary.",
        "casual": "Use an informal and relaxed tone with colloquial expressions.",
        "enthusiastic": "Use an energetic and engaging tone to captivate the audience.",
        "serious": "Use a solemn and focused tone suitable for important topics.",
        "humorous": "Incorporate light humor and wit to entertain while informing.",
        "gen-z": "Use Gen Z slang, memes, and internet terminology. Keep it casual, fast-paced, full of emojis, and TikTok-era references to resonate with younger audiences.",
        "technical": "Use precise, detailed, and domain-specific language tailored for researchers, engineers, and technical experts. Include terminology, data, and depth suitable for expert audiences."
    }

    length_guides = {
        "short": "Keep the transcript concise and to the point, avoiding unnecessary details. Keep the reading time between 2 to 5 minutes.",
        "medium": "Provide a balanced length with enough detail to be informative and engaging. Keep the reading time between 5 to 10 minutes.",
        "long": "Include comprehensive details and extended dialogue for depth."
    }

    def safe_json_loads(raw: str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            import re
            # Cut off at the last closing brace
            last_brace = raw.rfind("}")
            if last_brace != -1:
                raw_fixed = raw[: last_brace + 1]
                try:
                    return json.loads(raw_fixed)
                except Exception:
                    pass
            # fallback
            return {"transcript": []}

    guide_text = f"""
### FORMAT, STYLE & LENGTH GUIDES
- Format guidance: {format_guides.get(format_type, '')}
- Style guidance: {style_guides.get(style, '')}
- Length guidance: {length_guides.get(length, '')}
"""

    system_prompt = f"""You are the world-class {format_type} writer, you have worked as a ghostwriter for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris.
You are an expert content creator. Transform the provided text into a {format_type} 
that matches the requested audio style.
{guide_text}

========================
### RULES
- At the beginning of every audio, Speaker 1 must introduce the listener this this audio and the topic that will be talked about.
- The conversation should sound natural. Filler sounds like "Hmm", "Ahh", "Umm", "Oh", "Yeah", "Haha", "Hehe", "Wow" can appear, but very rarely and only when it feels absolutely natural. Avoid frequent or exaggerated use.

========================
### SPEAKER RULES
- Use between 1 and 6 speakers depending on the format.
- Speakers must be labeled "Speaker 1", "Speaker 2", ... up to "Speaker 6".
- Do not invent or assign real names, only use generic speaker labels.
- If {format_type} is narration, summary, article, or lecture: use at least 1 speaker.
- If {format_type} is podcast, interview, q-and-a, or tutorial: use at least 2 speakers.
- If {format_type} is panel-discussion, debate, meeting, or analysis: use at least 3 speakers (and up to 6).
"""
    if num_speakers is not None:
        system_prompt += f"""
========================
### SPEAKER OVERRIDE
- You MUST use exactly {num_speakers} speakers.
- Speakers must be labeled "Speaker 1" through "Speaker {num_speakers}".
- Do not add or remove speakers beyond this exact count.
"""

    system_prompt += """
========================
### CUSTOM PREFERENCES
"""
    if custom_preferences is not None:
        system_prompt += f"- Apply the following user preferences: {custom_preferences}\n"

    system_prompt += f"""
========================
### OUTPUT RULES
- Always output a **valid JSON object** with a top-level key "transcript".
- "transcript" must be a list of objects with two keys: "speaker" and "text".
- Each object = one speaker turn only.
- No latex, markdown, no commentary, meta-text, explanations, no titles.
- Do NOT include emojis, symbols, or non-speech artifacts.
- your speaker texts are turned directly into Speech using a Text-To-Speak model.

Example:
{{
  "transcript": [
    {{ "speaker": "Speaker 1", "text": "Welcome to Compu-Talk, your internal AI generated {format_type}!" }},
    {{ "speaker": "Speaker 2", "text": "Glad to be here." }},
    {{ "speaker": "Speaker N", "text": "Glad to be here too." }}
  ]
}}

========================
### CONTENT RULES
- Write in {language}.
- Match the conversational style of a real {format_type}.
- Allow interruptions, natural flow, or distinct voices when multiple speakers.
- Make it engaging, natural, and appropriate to the chosen format.
"""
    if is_first:
        system_prompt += "\n- Include an opening greeting appropriate to the format."
    if is_last:
        system_prompt += "\n- Include a closing goodbye appropriate to the format."
    if not is_first and not is_last:
        system_prompt += "\n- Do not include greetings or closings in this chunk."

    logger.info("Sending request to LLM for transcript generation...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.6,
        max_tokens=16384,
        response_format={"type": "json_object"},
        stop=["] }"],
    )

    raw = response.choices[0].message.content

    try:
        result = safe_json_loads(raw)
        if isinstance(result, dict) and "transcript" not in result:
            if "closing" in result:
                result = {"transcript": [{"speaker": "Speaker 1", "text": result["closing"]}]}
            elif "opening" in result:
                result = {"transcript": [{"speaker": "Speaker 1", "text": result["opening"]}]}
            elif "error" in result:
                logger.warning(f"LLM returned error instead of transcript: {result['error']}")
                result = {"transcript": [{"speaker": "Speaker 1", "text": f"[Error note: {result['error']}]"}]}
        if not isinstance(result, dict) or "transcript" not in result:
            raise ValueError("Result missing 'transcript' key")
        transcript_list = result["transcript"]
        if not isinstance(transcript_list, list):
            raise ValueError("'transcript' is not a list")
        output = []
        for entry in transcript_list:
            if not (isinstance(entry, dict) and "speaker" in entry and "text" in entry):
                raise ValueError(f"Invalid transcript entry: {entry}")
            output.append((entry["speaker"], entry["text"]))
        logger.info("Successfully parsed transcript from LLM response.")
        return output
    except Exception as e:
        logger.error(f"Transcript parsing failed: {e}\nRaw output: {raw[:300]}")
        raise

# ========== CHARACTER GENERATION ==========
def generate_characters(
    client: OpenAI,
    model: str,
    format_type: str,
    num_speakers: Optional[int] = None,
    custom_preferences: Optional[str] = None
) -> list:
    """
    Generate a list of character descriptions for the transcript, based on the number of speakers and preferences.
    """
    system_prompt = f"You are an expert at creating engaging, realistic podcast or discussion characters for audio. "
    if num_speakers is not None:
        system_prompt += f"Create exactly {num_speakers} distinct speakers for this {format_type}. "
    else:
        system_prompt += "Create a suitable number of distinct speakers for this format. "
    if custom_preferences is not None:
        system_prompt += f"Incorporate the following preferences into the character creation: {custom_preferences} "
    system_prompt += (
        "For each speaker, provide a short description of their persona, expertise, and speaking style. "
        "Output a JSON list with one object per speaker, each with keys 'speaker', 'persona', 'expertise', 'style'."
    )
    user_prompt = "Generate the list of speakers for the transcript."
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_tokens=2048,
        response_format={"type": "json_object"},
    )
    import json
    content = response.choices[0].message.content
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
        elif isinstance(result, dict) and "speakers" in result:
            return result["speakers"]
        else:
            raise ValueError("Unexpected response format for characters.")
    except Exception as e:
        logger.error(f"Failed to parse character list: {e}\nRaw: {content[:300]}")
        raise

# ========== TTS ==========
def generate_tts_audio(
        transcript: List[Tuple[str, str]],
        client: OpenAI,
        voices: dict,
        output_dir: Path,
        tts_model: str = "kokoro",
        format: str = "wav"
    ) -> str:
    """
    Generate TTS audio from a transcript and return the path to the final audio file.
    """
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    audio_segments = []

    for i, (speaker, text) in enumerate(transcript, 1):
        voice = voices.get(speaker, voices.get("default", "alloy"))
        logger.debug(f"Voice mapping for {speaker}: {voice}")
        out_file = segments_dir / f"segment_{i}.{format}"
        logger.info(f"Generating audio segment {i} for {speaker}...")
        resp = client.audio.speech.create(
            model=tts_model,
            voice=voice,
            input=text,
        )
        with open(out_file, "wb") as f:
            f.write(resp.read())
        logger.info(f"Audio segment {i} written to {out_file}")

        data, sr = sf.read(out_file)
        audio_segments.append(data)

    audio_audio = np.concatenate(audio_segments)
    final_path = output_dir / f"audio.{format}"
    sf.write(final_path, audio_audio, sr)
    logger.info(f"All audio segments concatenated and final audio saved to {final_path}")
    return str(final_path)

# ========== MAIN ==========
def generate_audio(
    pdf_path: str,
    output_dir: str = "./output",
    llm_model: str = "qwen3:30b-a3b-instruct-2507-q4_K_M",
    language: str = "english",
    format_type: Literal[
        "podcast", "narration", "interview", "panel-discussion", "summary", "article", "lecture",
        "q-and-a", "tutorial", "debate", "meeting", "analysis"
    ] = "podcast",
    style: Literal[
        "normal", "formal", "casual", "enthusiastic", "serious", "humorous", "gen-z", "technical"
    ] = "normal",
    length: Literal["short", "medium", "long"] = "medium",
    num_speakers: Optional[int] = None,
    custom_preferences: Optional[str] = None,
    transcript_file: Optional[str] = None,
):
    logger.info(f"generate_audio called with parameters: pdf_path={pdf_path}, output_dir={output_dir}, llm_model={llm_model}, language={language}, format_type={format_type}, style={style}, length={length}, num_speakers={num_speakers}, custom_preferences={custom_preferences}, transcript_file={transcript_file}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")  # Ollama API
    kokoro_client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")  # Kokoro API

    transcript = None
    transcript_json_path = output_dir / "transcript.json"
    # Step 1: Load transcript from file if provided, else generate
    if transcript_file is not None and Path(transcript_file).exists():
        logger.info(f"Loading transcript from JSON file: {transcript_file}")
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
            # Expecting a list of dicts with "speaker" and "text"
            transcript = [(entry["speaker"], entry["text"]) for entry in transcript_data]
    else:
        logger.info("Step 1 & 2: Extracting PDF text and generating transcript with LLM...")
        try:
            transcript = generate_transcript_from_pdf(
                pdf_path, ollama_client, llm_model, language,
                format_type=format_type, style=style, length=length,
                num_speakers=num_speakers, custom_preferences=custom_preferences
            )
            # Save as .txt
            transcript_path = output_dir / "transcript.txt"
            with open(transcript_path, "w", encoding="utf-8") as f:
                for speaker, text in transcript:
                    f.write(f"{speaker}: {text}\n")
            logger.info(f"Transcript saved to {transcript_path}")
            # Save as .json (list of dicts)
            transcript_json = [{"speaker": speaker, "text": text} for speaker, text in transcript]
            with open(transcript_json_path, "w", encoding="utf-8") as f:
                json.dump(transcript_json, f, ensure_ascii=False, indent=2)
            logger.info(f"Transcript JSON saved to {transcript_json_path}")
        except Exception as e:
            import traceback
            error_json_path = output_dir / "transcript_error.json"
            error_info = {
                "error": "Transcript generation failed",
                "details": str(e),
                "traceback": traceback.format_exc()
            }
            with open(error_json_path, "w", encoding="utf-8") as f:
                json.dump(error_info, f, ensure_ascii=False, indent=2)
            logger.error(f"Transcript generation failed. Error details written to {error_json_path}")
            raise

    logger.info("Step 3: Generating TTS audio...")
    voices = {
        "Speaker 1": "af_bella(1.4)+af_sky(0.8)",
        "Speaker 2": "am_michael+am_fenrir",
        "Speaker 4": "af_aoede(1)+af_kore(1)+af_sky(1.6)",
        "Speaker 3": "am_echo",
        "Speaker 6": "af_nova+af_jadzia",
        "Speaker 5": "am_adam",
        "default": "af_nova"
    }
    audio_file = generate_tts_audio(transcript, kokoro_client, voices, output_dir)

    logger.info(f"✅ Audio generated at: {audio_file}")