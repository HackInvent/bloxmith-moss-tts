"""Offline, killable MOSS-TTS-v1.5 adapter; verified upstream code, no vendored weights."""

from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import resource
import sys
import types
import wave

if __package__:
    from .assets_store import AssetError, prepare
    from .logic import configuration, LANGUAGES
    from .process import MossTtsError
else:
    package = types.ModuleType("moss_worker_owned")
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[package.__name__] = package
    from moss_worker_owned.assets_store import AssetError, prepare
    from moss_worker_owned.logic import configuration, LANGUAGES
    from moss_worker_owned.process import MossTtsError

MODEL_PARAMETERS = 8489841664
CODEC_PARAMETERS = 1774566400
RATE = 24000
STRIDE = 1920
NQ = 32
PAD = 1024
START, END, GEN, DELAY, EOS = 151652, 151653, 151656, 151662, 151645


def available_ram():
    values = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    available = int(values["MemAvailable"].split()[0]) * 1024
    try:
        cap = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if cap != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            available = min(available, max(0, int(cap) - used))
    except (OSError, ValueError):
        pass
    return available


def require_dependencies():
    from importlib.metadata import version
    if sys.version_info[:2] != (3, 12):
        raise MossTtsError("dependency")
    for name, expected in (("torch", "2.9.1"), ("torchaudio", "2.9.1"), ("torchcodec", "0.8.1"),
            ("transformers", "5.0.0"), ("accelerate", "1.10.1"), ("safetensors", "0.6.2"),
            ("numpy", "2.1.0"), ("einops", "0.8.1"), ("librosa", "0.11.0")):
        if version(name).split("+")[0] != expected:
            raise MossTtsError("dependency")


def resource_preflight(torch, cfg):
    """Conservative weight-plus-margin gate, not a RAM/VRAM reservation or performance claim."""
    weights = MODEL_PARAMETERS * (2 if cfg["precision"] == "bfloat16" else 4) + CODEC_PARAMETERS * 4
    required_ram = cfg["min_available_ram_mib"] * 1048576
    if cfg["device"] == "cpu":
        required_ram = max(required_ram, weights + 4 * 1073741824)
    else:
        if not torch.cuda.is_available() or cfg["precision"] == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise MossTtsError("device")
        free, _ = torch.cuda.mem_get_info(0)
        if free < weights + 2 * 1073741824:
            raise MossTtsError("resources")
    if available_ram() < required_ram:
        raise MossTtsError("resources")


def load_model(assets, cfg, job):
    """Called only after complete snapshot verification; local code runs in this owned process."""
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    dtype = torch.bfloat16 if cfg["precision"] == "bfloat16" else torch.float32
    # The pinned processor.from_pretrained forwards loader kwargs into its
    # constructor, but Transformers 5.0 rejects them there. Construct its exact
    # verified class with separately loaded LOCAL components, no SDK patching.
    common = {"trust_remote_code": True, "local_files_only": True}
    processor_class = get_class_from_dynamic_module("processing_moss_tts.MossTTSDelayProcessor",
        str(assets / "model"), local_files_only=True)
    config = AutoConfig.from_pretrained(str(assets / "model"), **common)
    tokenizer = AutoTokenizer.from_pretrained(str(assets / "model"), **common)
    codec = AutoModel.from_pretrained(str(assets / "codec"), **common,
        use_safetensors=True, dtype=torch.float32).to(cfg["device"]).eval()
    processor = processor_class(tokenizer=tokenizer, audio_tokenizer=codec, model_config=config)
    model = AutoModel.from_pretrained(str(assets / "model"), trust_remote_code=True,
        local_files_only=True, use_safetensors=True, dtype=dtype,
        attn_implementation="sdpa" if cfg["device"] == "cuda:0" else "eager").to(cfg["device"]).eval()
    expected = {"n_vq": NQ, "audio_vocab_size": PAD, "audio_pad_code": PAD,
        "audio_start_token_id": START, "audio_end_token_id": END, "im_end_token_id": EOS,
        "audio_assistant_gen_slot_token_id": GEN, "audio_assistant_delay_slot_token_id": DELAY,
        "sampling_rate": RATE}
    if any(getattr(model.config, key, None) != value or getattr(processor.model_config, key, None) != value
           for key, value in expected.items()):
        raise MossTtsError("assets_corrupt")
    codec = processor.audio_tokenizer.config
    if codec.sampling_rate != RATE or codec.downsample_rate != STRIDE:
        raise MossTtsError("assets_corrupt")
    return model, processor


def completed_codes(outputs, processor, cfg):
    """Validate complete assistant/audio boundaries and RVQ padding before any codec call."""
    import numpy as np
    if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], tuple) or len(outputs[0]) != 2:
        raise MossTtsError("audio_invalid")
    start, codes = outputs[0]
    # Generation mode has no assistant prefix to crop. Continuation is not requested.
    prefix = start.item() if hasattr(start, "item") else start
    if type(prefix) is not int or prefix != 0 or getattr(codes, "ndim", None) != 2 or codes.shape[1] != NQ + 1:
        raise MossTtsError("audio_invalid")
    raw = codes.detach().cpu().numpy()
    if raw.dtype.kind not in "iu" or not 1 <= raw.shape[0] <= cfg["max_new_tokens"]:
        raise MossTtsError("audio_invalid")
    if raw[-1, 0] != EOS or np.any(raw[:-1, 0] == EOS):
        raise MossTtsError("audio_limit")
    if np.any(raw[:, 0] < 0) or np.any(raw[:, 0] >= 155648):
        raise MossTtsError("audio_invalid")
    regions, opened = [], None
    total_frames = 0
    for index, row in enumerate(raw):
        token = int(row[0])
        if token == START:
            if opened is not None or not np.all(row[1:] == PAD):
                raise MossTtsError("audio_invalid")
            opened = index + 1
        elif token == END:
            if opened is None or not np.all(row[1:] == PAD):
                raise MossTtsError("audio_invalid")
            delayed = raw[opened:index, 1:]
            frames = delayed.shape[0] - NQ + 1
            if frames < 1:
                raise MossTtsError("audio_invalid")
            total_frames += frames
            if len(regions) >= cfg["max_segments"] or total_frames * STRIDE > cfg["max_audio_sec"] * RATE:
                raise MossTtsError("audio_limit")
            # A true delayed RVQ stream has exactly this prefix/tail padding.
            for channel in range(NQ):
                column = delayed[:, channel]
                actual = column[channel:channel + frames]
                if np.any(column[:channel] != PAD) or np.any(column[channel + frames:] != PAD) or np.any(actual < 0) or np.any(actual >= PAD):
                    raise MossTtsError("audio_invalid")
            decoded_codes = processor.apply_de_delay_pattern(codes[opened:index, 1:])
            if tuple(decoded_codes.shape) != (frames, NQ):
                raise MossTtsError("audio_invalid")
            regions.append(decoded_codes)
            opened = None
        elif opened is not None:
            if token not in (GEN, DELAY):
                raise MossTtsError("audio_invalid")
        elif token in (GEN, DELAY, 151644) or not np.all(row[1:] == PAD):
            raise MossTtsError("audio_invalid")
    if opened is not None or not regions:
        raise MossTtsError("audio_limit")
    return regions, raw.shape[0]


def generate_audio(model, processor, text, cfg, job):
    import numpy as np
    message = processor.build_user_message(text=text,
        reference=[str(job / "reference.wav")] if cfg["reference_audio"] else None,
        language=LANGUAGES.get(cfg["language"]),
        tokens=round(cfg["target_duration_sec"] * 12.5) if cfg["target_duration_sec"] else None)
    batch = processor([[message]], mode="generation")
    ids, mask = batch["input_ids"].to(cfg["device"]), batch["attention_mask"].to(cfg["device"])
    if ids.ndim != 3 or ids.shape[0] != 1 or ids.shape[2] != NQ + 1 or tuple(mask.shape) != tuple(ids.shape[:2]):
        raise MossTtsError("model_failed")
    if ids.shape[1] + cfg["max_new_tokens"] > model.config.language_config.max_position_embeddings:
        raise MossTtsError("audio_limit")
    outputs = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=cfg["max_new_tokens"],
        **{key: cfg[key] for key in ("text_temperature", "text_top_p", "text_top_k",
            "audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty")})
    regions, tokens = completed_codes(outputs, processor, cfg)
    audio, clipped = [], 0
    for region in regions:
        # Codec streaming decode is batch-size 1. Keep each complete region, in order;
        # do not use the upstream parser's multi-region split or batch them together.
        decoded = processor.decode_audio_codes([region])
        if not isinstance(decoded, list) or len(decoded) != 1:
            raise MossTtsError("audio_invalid")
        samples = decoded[0].detach().cpu().numpy()
        if samples.ndim != 1 or samples.dtype.kind != "f" or len(samples) != region.shape[0] * STRIDE or not np.all(np.isfinite(samples)):
            raise MossTtsError("audio_invalid")
        if np.max(np.abs(samples)) > 4:
            raise MossTtsError("audio_invalid")
        clipped += int(np.count_nonzero(np.abs(samples) > 1))
        audio.append(np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2"))
    pcm = np.concatenate(audio)
    if not 0 < len(pcm) <= cfg["max_audio_sec"] * RATE or not np.any(pcm):
        raise MossTtsError("audio_invalid")
    path = job / "generated.wav"
    with path.open("xb") as target:
        os.chmod(path, 0o600)
        with wave.open(target, "wb") as wav:
            wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(RATE); wav.writeframes(pcm.tobytes())
    return {"complete": True, "eos": True, "tokens": tokens, "segments": len(regions),
        "sample_rate": RATE, "frames": len(pcm), "clipped_samples": clipped}


def infer(item):
    cfg = configuration(item["config"])
    if cfg["memory_limit_mib"]:
        limit = cfg["memory_limit_mib"] * 1048576
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cpu_limit = cfg["timeout_sec"] * cfg["threads"]
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit + 1))
    require_dependencies()
    import torch
    torch.set_num_threads(cfg["threads"]); torch.set_num_interop_threads(1)
    resource_preflight(torch, cfg)
    if cfg["device"] == "cuda:0":
        torch.backends.cuda.enable_cudnn_sdp(False)
    assets = prepare(item["assets"])
    job = Path(item["job"])
    model, processor = load_model(assets, cfg, job)
    with torch.inference_mode():
        return generate_audio(model, processor, item["request"]["text"], cfg, job)


def main():
    data = sys.stdin.buffer.read(32769)
    if len(data) > 32768:
        raise MossTtsError("input")
    with redirect_stdout(sys.stderr):
        result = infer(json.loads(data))
    sys.stdout.write(json.dumps(result, ensure_ascii=True, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        from importlib.metadata import PackageNotFoundError
        if isinstance(exc, MossTtsError): code = exc.code
        elif isinstance(exc, AssetError): code = str(exc)
        elif isinstance(exc, (ImportError, PackageNotFoundError)): code = "dependency"
        elif isinstance(exc, MemoryError) or isinstance(exc, RuntimeError) and any(s in str(exc) for s in ("out of memory", "DefaultCPUAllocator", "Cannot allocate memory")):
            code = "resources"
        else: code = "model_failed"
        print(json.dumps({"error_code": code})); raise SystemExit(1) from None
