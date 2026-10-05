"""Explicit native-model doubles; genuine delay protocol, owned adapter and codecs tested."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import wave
import numpy as np
from blocs.moss_tts import logic, worker
from moss_fixture import settings


class Tensor:
    """Small CPU-only tensor double, not a replacement inference engine."""
    def __init__(self, data): self.data = np.asarray(data)
    @property
    def shape(self): return self.data.shape
    @property
    def ndim(self): return self.data.ndim
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.data
    def to(self, *args, **kwargs): return self
    def __getitem__(self, key): return Tensor(self.data[key])


def output(lengths=(3, 5)):
    """Original fixture implementing the published 32-codebook delay layout."""
    rows = []
    for number, length in enumerate(lengths):
        rows.append([worker.START] + [worker.PAD] * worker.NQ)
        delayed = np.full((length + 31, 32), worker.PAD, dtype=np.int64)
        for channel in range(32):
            delayed[channel:channel + length, channel] = np.arange(length) + number * 10 + channel
        for step, data in enumerate(delayed):
            # Actual generation finishes with 32 delay slots; first delay still
            # carries the last first-codebook value before channel-tail draining.
            rows.append([worker.GEN if step < length - 1 else worker.DELAY] + data.tolist())
        rows.append([worker.END] + [worker.PAD] * 32)
        rows.append([198] + [worker.PAD] * 32)
    rows.append([worker.EOS] + [worker.PAD] * 32)
    return [(0, Tensor(np.array(rows, dtype=np.int64)))]


class Processor:
    def __init__(self): self.decode_calls = []
    def build_user_message(self, **kw): self.message = kw; return kw
    def __call__(self, conversations, **kw):
        self.conversations, self.options = conversations, kw
        return {"input_ids": Tensor(np.zeros((1, 20, 33), dtype=np.int64)),
            "attention_mask": Tensor(np.ones((1, 20), dtype=bool))}
    def apply_de_delay_pattern(self, codes):
        data = codes.numpy(); frames = data.shape[0] - 31
        return Tensor(np.stack([data[i:i + frames, i] for i in range(32)], axis=1))
    def decode_audio_codes(self, regions):
        assert len(regions) == 1, "Official chunked codec only supports one item"
        self.decode_calls.append(regions)
        codes = regions[0].numpy()
        # Opposite signs distinguish regions and reveal dropped/reordered audio.
        return [Tensor(np.full(codes.shape[0] * 1920, .2 if codes[0, 0] == 0 else -.3, dtype=np.float32))]


class Model:
    def __init__(self, outputs=None):
        self.outputs = outputs if outputs is not None else output()
        self.config = NS(language_config=NS(max_position_embeddings=40960))
    def generate(self, **kw): self.arguments = kw; return self.outputs


class WorkerTests(unittest.TestCase):
    def test_configuration_matches_fields_and_languages(self):
        package = Path(__file__).parents[1]
        fields = json.loads((package / "fields.json").read_text())
        self.assertEqual({f["key"] for f in fields}, set(logic.DEFAULTS))
        self.assertEqual(json.loads((package / "model.json").read_text())["config"], logic.DEFAULTS)
        self.assertEqual(len(logic.LANGUAGES), 31)
        for field in fields:
            if field["type"] in {"integer", "number"}:
                for value in (True, None, "1", float("nan"), field["min"] - 1, field["max"] + 1):
                    with self.subTest(key=field["key"], value=value), self.assertRaises(logic.MossTtsError):
                        logic.configuration({field["key"]: value})
        for cfg in ({"language": "fake"}, {"precision": "float16"}, {"device": "cpu"},
                {"reference_audio": "\ud800"}, {"memory_limit_mib": 4095}, {"target_duration_sec": .01},
                {"target_duration_sec": 3, "max_audio_sec": 2}):
            with self.subTest(cfg=cfg), self.assertRaises(logic.MossTtsError): logic.configuration(cfg)

    def test_complete_regions_and_native_arguments(self):
        cfg = logic.configuration({"language": "fr", "target_duration_sec": 2})
        model, processor = Model(), Processor()
        with TemporaryDirectory() as directory:
            job = Path(directory)
            info = worker.generate_audio(model, processor, "Bonjour [pause 0.2s] !", cfg, job)
            self.assertEqual(info["segments"], 2)
            self.assertEqual(info["frames"], 8 * 1920)
            self.assertEqual(logic.audio_info(job / "generated.wav", cfg, json.dumps(info)), info)
            with wave.open(str(job / "generated.wav")) as wav:
                pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            self.assertTrue(np.all(pcm[:3 * 1920] > 0)); self.assertTrue(np.all(pcm[3 * 1920:] < 0))
        self.assertEqual(processor.message, {"text": "Bonjour [pause 0.2s] !", "reference": None, "language": "French", "tokens": 25})
        self.assertEqual(processor.options, {"mode": "generation"})
        self.assertEqual(len(processor.decode_calls), 2)
        self.assertEqual(set(model.arguments), {"input_ids", "attention_mask", "max_new_tokens", "text_temperature", "text_top_p", "text_top_k", "audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty"})
        for key in set(model.arguments) - {"input_ids", "attention_mask"}: self.assertEqual(model.arguments[key], cfg[key])

    def test_optional_reference_is_explicit(self):
        cfg = logic.configuration({"reference_audio": "selected.wav"})
        with TemporaryDirectory() as directory:
            job = Path(directory); processor = Processor()
            worker.generate_audio(Model(), processor, "Hello", cfg, job)
            self.assertEqual(processor.message["reference"], [str(job / "reference.wav")])
            self.assertIsNone(processor.message["language"]); self.assertIsNone(processor.message["tokens"])

    def test_incomplete_or_corrupt_protocol_never_decodes(self):
        def mutated(change):
            outputs = output((3,)); change(outputs[0][1].data); return outputs
        cases = [[], [(1, output()[0][1])], [(False, output()[0][1])],
            mutated(lambda a: a.__setitem__((-1, 0), 198)),
            mutated(lambda a: a.__setitem__((2, 0), worker.EOS)),
            mutated(lambda a: a.__setitem__((1, 1), worker.PAD)),
            mutated(lambda a: a.__setitem__((1, 32), 5)),
            mutated(lambda a: a.__setitem__((1, 1), -1)),
            mutated(lambda a: a.__setitem__((-3, 0), 198)),
            mutated(lambda a: a.__setitem__((-2, 0), worker.DELAY)),
            mutated(lambda a: a.__setitem__((-1, 1), 7)),
            [(0, Tensor(np.ones((2, 32), dtype=np.int64)))],
            [(0, Tensor(np.ones((2, 33), dtype=np.float32)))]]
        for outputs in cases:
            processor = Processor()
            with TemporaryDirectory() as directory, self.subTest(outputs=outputs), self.assertRaises(logic.MossTtsError):
                worker.generate_audio(Model(outputs), processor, "Hello", logic.DEFAULTS, Path(directory))
            self.assertFalse(processor.decode_calls)

    def test_all_limits_precede_codec_work(self):
        for outputs, cfg in ((output((13,)), {"max_audio_sec": 1}), (output(), {"max_segments": 1}),
                (output((40,)), {"max_new_tokens": 64})):
            processor = Processor()
            with TemporaryDirectory() as directory, self.assertRaises(logic.MossTtsError):
                worker.generate_audio(Model(outputs), processor, "Hello", logic.configuration(cfg), Path(directory))
            self.assertFalse(processor.decode_calls)

    def test_invalid_waveforms_and_pcm_clipping(self):
        for audio in (np.zeros(5760), np.full(5760, np.nan), np.full(5760, 5.), np.ones(2), np.ones((5760, 1))):
            processor = Processor(); processor.decode_audio_codes = lambda _: [Tensor(audio)]
            with TemporaryDirectory() as directory, self.subTest(shape=audio.shape), self.assertRaises(logic.MossTtsError):
                worker.generate_audio(Model(output((3,))), processor, "Hello", logic.DEFAULTS, Path(directory))
        processor = Processor(); processor.decode_audio_codes = lambda _: [Tensor(np.full(5760, 1.1))]
        with TemporaryDirectory() as directory:
            info = worker.generate_audio(Model(output((3,))), processor, "Hello", logic.DEFAULTS, Path(directory))
            self.assertEqual(info["clipped_samples"], 5760)

    def test_local_component_loader_does_not_leak_kwargs_to_processor(self):
        config = NS(n_vq=32, audio_vocab_size=1024, audio_pad_code=1024, audio_start_token_id=worker.START,
            audio_end_token_id=worker.END, im_end_token_id=worker.EOS, audio_assistant_gen_slot_token_id=worker.GEN,
            audio_assistant_delay_slot_token_id=worker.DELAY, sampling_rate=24000)
        calls = []
        class Native:
            def __init__(self, cfg): self.config = cfg
            def to(self, device): self.device = device; return self
            def eval(self): return self
        def model(path, **kw):
            calls.append((path, kw))
            return Native(NS(sampling_rate=24000, downsample_rate=1920) if path.endswith("codec") else config)
        def processor(*, tokenizer, audio_tokenizer, model_config):
            return NS(tokenizer=tokenizer, audio_tokenizer=audio_tokenizer, model_config=model_config)
        def component(path, **kw):
            calls.append((path, kw)); return config
        dynamic_calls = []
        def dynamic(*args, **kw): dynamic_calls.append((args, kw)); return processor
        with TemporaryDirectory() as directory, patch.dict(sys.modules, {
                "torch": NS(float32="fp32", bfloat16="bf16"),
                "transformers": NS(AutoModel=NS(from_pretrained=model), AutoConfig=NS(from_pretrained=component), AutoTokenizer=NS(from_pretrained=component)),
                "transformers.dynamic_module_utils": NS(get_class_from_dynamic_module=dynamic)}):
            root = Path(directory)
            native, processor = worker.load_model(root, logic.DEFAULTS, root)
            self.assertEqual(native.device, "cuda:0"); self.assertEqual(processor.audio_tokenizer.device, "cuda:0")
            self.assertEqual(len(calls), 4)
            self.assertTrue(all(c[0] in {str(root / "model"), str(root / "codec")} and c[1]["local_files_only"] is True for c in calls))
            self.assertEqual(calls[2][1]["dtype"], "fp32"); self.assertEqual(calls[3][1]["dtype"], "bf16")
            self.assertEqual(calls[3][1]["attn_implementation"], "sdpa")
            self.assertEqual(dynamic_calls, [(("processing_moss_tts.MossTTSDelayProcessor", str(root / "model")), {"local_files_only": True})])

    def test_resource_preflight(self):
        torch = NS(cuda=NS(is_available=lambda: True, is_bf16_supported=lambda: True, mem_get_info=lambda _: (64 * 1073741824, 64 * 1073741824)))
        with patch.object(worker, "available_ram", return_value=64 * 1073741824):
            worker.resource_preflight(torch, logic.DEFAULTS)
            worker.resource_preflight(torch, logic.configuration({"device": "cpu", "precision": "float32"}))
        with patch.object(worker, "available_ram", return_value=32 * 1073741824), self.assertRaisesRegex(logic.MossTtsError, "RAM"):
            worker.resource_preflight(torch, logic.configuration({"device": "cpu", "precision": "float32"}))
        torch.cuda.mem_get_info = lambda _: (8 * 1073741824, 8 * 1073741824)
        with self.assertRaises(logic.MossTtsError): worker.resource_preflight(torch, logic.DEFAULTS)
        torch.cuda.is_available = lambda: False
        with self.assertRaises(logic.MossTtsError): worker.resource_preflight(torch, logic.DEFAULTS)

    def test_reference_limits_and_preservation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory); cfg = logic.configuration(settings(root)); source = Path(cfg["reference_audio"])
            for rate, frames, accepted in ((8000, 8000, True), (16000, 480000, True), (24000, 720000, True),
                    (24000, 720001, False), (44100, 1323000, True), (48000, 1440000, True),
                    (48000, 1444800, False), (24000, 100, False)):
                samples = (np.sin(np.arange(frames) * 440 * 2 * np.pi / rate) * 8000).astype("<i2")
                with wave.open(str(source), "wb") as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(samples.tobytes())
                before = source.read_bytes()
                with TemporaryDirectory(dir=root) as temporary:
                    try: result = logic.prepare_reference(cfg, root, Path(temporary), time.monotonic() + 10, lambda: False)
                    except logic.MossTtsError as exc: self.assertFalse(accepted, (rate, frames, exc))
                    else: self.assertTrue(accepted, (rate, frames, result)); self.assertLessEqual(result["duration_sec"], 30)
                self.assertEqual(source.read_bytes(), before)
            with TemporaryDirectory(dir=root) as temporary:
                job = Path(temporary)
                self.assertIsNone(logic.prepare_reference({**cfg, "reference_audio": ""}, root, job, time.monotonic() + 1, lambda: False))
                self.assertFalse(list(job.iterdir()))
            with TemporaryDirectory(dir=root) as temporary, self.assertRaises(logic.MossTtsError):
                logic.prepare_reference({**cfg, "reference_audio": str(root / "missing.wav")}, root, Path(temporary), time.monotonic() + 1, lambda: False)


if __name__ == "__main__": unittest.main()
