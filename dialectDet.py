from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

# ── Config ─────────────────────────────────────────────────────────────────────
# Using ASR-optimized variant — best for transcription + code-switching
MODEL_ID    = "MERaLiON/MERaLiON-2-10B-ASR"
SAMPLE_RATE = 16000

INSTRUCTION = (
    "Please transcribe this speech exactly as spoken, "
    "including any code-switching between English, Mandarin, Malay, or Tamil."
)

# ── Device detection ───────────────────────────────────────────────────────────
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")

# ── Load model ─────────────────────────────────────────────────────────────────
print("Loading MERaLiON-2 model... (this may take a few minutes)")

# MERaLiON-2 uses MERaLiON2ForConditionalGeneration (note the 2)
MERaLiON2Model = get_class_from_dynamic_module(
    "modeling_meralion2.MERaLiON2ForConditionalGeneration",
    MODEL_ID
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

dtype = torch.float16 if DEVICE != "cpu" else torch.float32
model = MERaLiON2Model.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=dtype,
).to(DEVICE)
model.eval()
print("Model loaded.\n")

# ── Transcribe a single audio chunk ───────────────────────────────────────────
def transcribe_chunk(audio_array: np.ndarray) -> str:
    inputs = processor(
        audio=audio_array,
        sampling_rate=SAMPLE_RATE,
        text=INSTRUCTION,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=300)

    return processor.decode(output_ids[0], skip_special_tokens=True).strip()

# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset(
    "MERaLiON/Multitask-National-Speech-Corpus-v1",
    data_dir="ASR-PART6-Test"  # 1k rows, heavy code-switching
)["train"]
print(f"Dataset loaded: {len(ds)} samples\n")

# ── Run on dataset ─────────────────────────────────────────────────────────────
for i, sample in enumerate(ds):
    audio_array = sample["audio"]["array"]  # already 16kHz numpy array
    ground_truth = sample["answer"]         # ground truth transcript

    prediction = transcribe_chunk(audio_array)

    print(f"── Sample {i+1} ──────────────────────────")
    print(f"Ground truth : {ground_truth}")
    print(f"Predicted    : {prediction}")
    print()

    if i == 9:  # remove to run on all 1k samples
        break