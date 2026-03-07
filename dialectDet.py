from datasets import load_dataset
import torch
import numpy as np
from transformers import AutoProcessor
from transformers.dynamic_module_utils import get_class_from_dynamic_module
import traceback
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "MERaLiON/MERaLiON-2-10B-ASR"
SAMPLE_RATE = 16000

PROMPT_TEMPLATE = "Instruction: {query} \nFollow the text instruction based on the following audio: <SpeechHere>"

INSTRUCTION = "Please transcribe this speech."

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

try:
    MERaLiON2Model = get_class_from_dynamic_module(
        "modeling_meralion2.MERaLiON2ForConditionalGeneration",
        MODEL_ID
    )
    print("✓ Model class loaded")

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("✓ Processor loaded")

    model = MERaLiON2Model.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_buffers=True,
    )
    model.eval()
    print("✓ Model loaded\n")

except Exception as e:
    print(f"\n❌ Error loading model: {e}")
    traceback.print_exc()
    exit()

# ── Transcribe a single audio chunk ───────────────────────────────────────────
def transcribe_chunk(audio_array: np.ndarray) -> str:
    audio_array = audio_array.astype(np.float32)

    # Build prompt using the official MERaLiON template
    prompt = PROMPT_TEMPLATE.format(query=INSTRUCTION)
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

    # Convert and move to GPU explicitly
    processed_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            tensor = v.to(DEVICE)
        else:
            tensor = torch.tensor(v).to(DEVICE)
        
        # Cast float32 to float16 to match model weights
        if tensor.dtype == torch.float32:
            tensor = tensor.to(torch.float16)
        
        processed_inputs[k] = tensor

    with torch.no_grad():
        output_ids = model.generate(**processed_inputs, max_new_tokens=300)

    # Strip the input tokens — only decode what the model generated
    input_length = processed_inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][input_length:]
    return processor.decode(generated_ids, skip_special_tokens=True).strip()
# ── Load dataset ───────────────────────────────────────────────────────────────
print("Loading dataset...")
ds = load_dataset(
    "MERaLiON/Multitask-National-Speech-Corpus-v1",
    data_dir="ASR-PART6-Test"
)["train"]
print(f"Dataset loaded: {len(ds)} samples\n")

# ── Run on first 5 samples ─────────────────────────────────────────────────────
for i, sample in enumerate(ds.select(range(5))):
    audio_array  = sample["context"]["array"]
    ground_truth = sample["answer"]

    if audio_array is None:
        print(f"── Sample {i+1} — skipped (no audio)\n")
        continue

    try:
        prediction = transcribe_chunk(audio_array)
        print(f"── Sample {i+1} ──────────────────────────")
        print(f"Ground truth : {ground_truth}")
        print(f"Predicted    : {prediction}")
        print()
    except Exception as e:
        print(f"── Sample {i+1} — transcription error: {e}")
        traceback.print_exc()
        print()