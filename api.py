import io
import traceback

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, BitsAndBytesConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "MERaLiON/MERaLiON-2-10B-ASR"
SAMPLE_RATE = 16000

PROMPT_TEMPLATE = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"

TRANSCRIBE_INSTRUCTION = "Please transcribe this speech."

TRANSLATE_INSTRUCTION = "Please translate the speech into English."

# ── Device detection ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ── Load model at startup ──────────────────────────────────────────────────────
print("Loading MERaLiON-2 model... (this may take a few minutes)")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

try:
    MERaLiON2Model = get_class_from_dynamic_module(
        "modeling_meralion2.MERaLiON2ForConditionalGeneration",
        MODEL_ID
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = MERaLiON2Model.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_buffers=True,
    )
    model.eval()
    print("✓ Model ready\n")
except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    traceback.print_exc()
    raise SystemExit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = 50
MAX_FILE_BYTES   = MAX_FILE_SIZE_MB * 1024 * 1024

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="MERaLiON Transcription API")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler — keeps the server alive on any unhandled exception."""
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


def run_inference(audio_array: np.ndarray, instruction: str) -> str:
    """Run model inference on a float32 audio array."""
    audio_array = audio_array.astype(np.float32)

    # Build prompt using the official MERaLiON template
    prompt = PROMPT_TEMPLATE.format(query=instruction)
    conversation = [[{"role": "user", "content": prompt}]]

    text_prompt = processor.tokenizer.apply_chat_template(
        conversation=conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text_prompt,
        audios=audio_array,
    )

    processed_inputs = {}
    for k, v in inputs.items():
        tensor = v.to(DEVICE) if isinstance(v, torch.Tensor) else torch.tensor(v).to(DEVICE)
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)
        processed_inputs[k] = tensor

    try:
        with torch.no_grad():
            output_ids = model.generate(**processed_inputs, max_new_tokens=300)
    except torch.cuda.OutOfMemoryError:
        # Free GPU cache so subsequent requests can still succeed
        torch.cuda.empty_cache()
        raise RuntimeError(
            "GPU out of memory. The audio clip may be too long. "
            "Try a shorter recording."
        )

    # Strip the input tokens — only decode what the model generated
    input_length = processed_inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_length:]
    result = processor.decode(generated_ids, skip_special_tokens=True).strip()
    return result


def load_audio(file_bytes: bytes) -> np.ndarray:
    """Load audio bytes, validate, and resample to 16 kHz mono if needed."""
    if len(file_bytes) == 0:
        raise ValueError("Uploaded file is empty.")

    try:
        audio, sr = sf.read(io.BytesIO(file_bytes))
    except Exception:
        raise ValueError(
            "Could not decode audio. Supported formats: wav, flac, ogg, aiff. "
            "For mp3/m4a, install ffmpeg."
        )

    if len(audio) == 0:
        raise ValueError("Audio file contains no samples.")

    # Convert stereo to mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if necessary
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)

    return audio.astype(np.float32)


@app.get("/")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Upload an audio file and receive a transcription.

    Accepts: wav, flac, ogg, aiff (mp3/m4a needs ffmpeg)
    Returns: { "transcription": "..." }
    """
    try:
        audio_bytes = await file.read()

        if len(audio_bytes) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
            )

        audio_array = load_audio(audio_bytes)
        result = run_inference(audio_array, TRANSCRIBE_INSTRUCTION)
        return JSONResponse({"transcription": result})
    except HTTPException:
        raise
    except ValueError as e:
        # Bad input — tell the client what was wrong
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # OOM or inference error — server still alive
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


@app.post("/translate")
async def translate(file: UploadFile = File(...)):
    """
    Upload an audio file and receive a transcription + English translation.

    Accepts: wav, flac, ogg, aiff (mp3/m4a needs ffmpeg)
    Returns: { "result": "..." }
    """
    try:
        audio_bytes = await file.read()

        if len(audio_bytes) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB} MB.",
            )

        audio_array = load_audio(audio_bytes)
        result = run_inference(audio_array, TRANSLATE_INSTRUCTION)
        return JSONResponse({"result": result})
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
