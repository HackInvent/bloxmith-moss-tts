"""Owned killable process supervision. No model imports or framework internals."""
from contextlib import suppress
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ERRORS = {
    "platform": "MOSS-TTS-v1.5 currently requires Linux.",
    "input": "Send bounded valid text or an explicit interrupt command.",
    "config": "Check the MOSS-TTS-v1.5 configuration.",
    "assets_missing": "Prepare the pinned MOSS-TTS-v1.5 model and all auxiliary model files.",
    "assets_corrupt": "A model asset failed integrity verification; it was not loaded.",
    "dependency": "The selected Python lacks the pinned MOSS inference dependencies.",
    "device": "The selected CUDA device or precision is unavailable.",
    "resources": "The available RAM or configured process memory limit is insufficient.",
    "model_failed": "Local MOSS synthesis failed; check the model prerequisites and resource limits.",
    "audio_limit": "Generation did not reach end-of-sequence; no partial audio was published.",
    "reference": "Choose a complete, non-silent local voice reference, between 0.5 and 30 seconds and at most 32 MiB.",
    "audio_invalid": "The generated audio is empty, incomplete or invalid.",
    "process_output": "A native process exceeded its bounded diagnostic output.",
    "timeout": "The request deadline expired; no incomplete audio was published.",
    "cancelled": "Synthesis cancelled; no incomplete audio was published.",
    "storage": "Check the block storage and output directories.",
    "queue_full": "The local pending queue is full; this request was not accepted.",
    "listener": "The synthesis listener is unavailable; Stop and reload the Run.",
}

class MossTtsError(ValueError):
    def __init__(self, code):
        self.code = code if code in ERRORS else "model_failed"
        super().__init__(ERRORS[self.code])

class Cancelled(MossTtsError):
    def __init__(self):
        super().__init__("cancelled")

def check(deadline, cancel):
    if cancel():
        raise Cancelled()
    if time.monotonic() >= deadline:
        raise MossTtsError("timeout")

def run_process(args, *, directory, deadline, cancel, limit, stdin=None):
    """Bound every pipe; fence children on cancellation and abrupt framework-host death."""
    check(deadline, cancel)
    if sys.platform != "linux":
        raise MossTtsError("platform")
    envelope = stdin or b""
    if len(envelope) > 32768:
        raise MossTtsError("input")
    environment = {k: v for k, v in os.environ.items() if k in {"PATH", "LANG", "LC_ALL"}}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1",
        HF_HOME=str(directory / "hf-cache"), HF_MODULES_CACHE=str(directory / "hf-modules"), XDG_CACHE_HOME=str(directory / "cache"), TORCH_HOME=str(directory / "torch-cache"),
        NUMBA_CACHE_DIR=str(directory / "numba-cache"), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2")
    command = [sys.executable, "-E", "-B", str(Path(__file__).with_name("process_guard.py")), str(os.getpid()), *args]
    process = subprocess.Popen(command, cwd=directory, env=environment, start_new_session=True,
        stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    selector, output, errors, offset = selectors.DefaultSelector(), bytearray(), bytearray(), 0
    try:
        if process.stdin is not None:
            os.set_blocking(process.stdin.fileno(), False)
            if envelope:
                selector.register(process.stdin, selectors.EVENT_WRITE, "in")
            else:
                process.stdin.close()
        for stream, name in ((process.stdout, "out"), (process.stderr, "err")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        while selector.get_map():
            check(deadline, cancel)
            for key, _ in selector.select(timeout=0.03):
                if key.data == "in":
                    try:
                        offset += os.write(key.fd, envelope[offset:offset + 4096])
                    except BrokenPipeError:
                        offset = len(envelope)
                    if offset == len(envelope):
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                target, maximum = (output, limit) if key.data == "out" else (errors, 262144)
                if len(target) + len(chunk) > maximum:
                    raise MossTtsError("process_output")
                target.extend(chunk)
        while process.poll() is None:
            check(deadline, cancel)
            time.sleep(0.02)
        check(deadline, cancel)
        if process.returncode:
            try:
                detail = json.loads(output).get("error_code")
            except (ValueError, AttributeError):
                detail = None
            raise MossTtsError(detail if detail in ERRORS else "model_failed")
        return bytes(output)
    finally:
        selector.close()
        # The guardian owns this group. Fence descendants even after normal exit.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
