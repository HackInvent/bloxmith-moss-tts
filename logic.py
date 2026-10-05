"""Bounded complete-file synthesis; publication is separate from cancellable native work."""

from collections.abc import Mapping
from dataclasses import dataclass
import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import time
from uuid import uuid4
import wave
from .process import MossTtsError, Cancelled, check, run_process

DEFAULTS = {
    "asset_directory": "",
    "environment_python": "",
    "reference_audio": "",
    "output_directory": "",
    "language": "auto",
    "device": "cuda:0",
    "precision": "bfloat16",
    "format": "ogg_opus",
    "target_duration_sec": 0,
    "text_temperature": 1.5,
    "text_top_p": 1,
    "text_top_k": 50,
    "audio_temperature": 1.7,
    "audio_top_p": 0.8,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1,
    "threads": 2,
    "max_audio_sec": 60,
    "max_new_tokens": 1024,
    "max_segments": 16,
    "max_text_chars": 1000,
    "timeout_sec": 240,
    "memory_limit_mib": 0,
    "min_available_ram_mib": 32768,
    "max_pending": 4
}
LANGUAGES = {
    "zh": "Chinese",
    "yue": "Cantonese",
    "en": "English",
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "fi": "Finnish",
    "fr": "French",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese"
}


def configuration(raw):
    if not isinstance(raw, Mapping) or set(raw)-set(DEFAULTS)-{"execution","position","runtime_path","runtime_path_label"}:
        raise MossTtsError("config")
    cfg={**DEFAULTS,**{k:v for k,v in raw.items() if k in DEFAULTS}}
    for key in ("asset_directory","environment_python","reference_audio","output_directory"):
        value=cfg[key]
        try:
            if not isinstance(value,str) or len(value.encode("utf-8"))>4096 or any(ord(c)<32 for c in value):
                raise MossTtsError("config")
        except UnicodeError:
            raise MossTtsError("config") from None
    for key,choices in (("device",("cpu","cuda:0")),("precision",("float32","bfloat16")),("format",("wav","ogg_opus")),("language",("auto",*LANGUAGES))):
        if not isinstance(cfg[key],str) or cfg[key] not in choices:
            raise MossTtsError("config")
    if cfg["device"]=="cpu" and cfg["precision"]!="float32":
        raise MossTtsError("config")
    for key,low,high in (("text_top_k",0,500),("audio_top_k",0,1024),("threads",1,16),("max_audio_sec",1,90),
            ("max_text_chars",1,2000),("timeout_sec",1,240),("memory_limit_mib",0,262144),
            ("min_available_ram_mib",8192,262144),("max_pending",0,16),("max_new_tokens",64,4096),("max_segments",1,16)):
        if type(cfg[key]) is not int or not low<=cfg[key]<=high:
            raise MossTtsError("config")
    if 0 < cfg["memory_limit_mib"] < 4096:
        raise MossTtsError("config")
    for key,low,high in (("text_temperature",0,2),("audio_temperature",0,2),("text_top_p",.01,1),("audio_top_p",.01,1),("audio_repetition_penalty",1,2),("target_duration_sec",0,90)):
        if type(cfg[key]) not in (int,float) or not math.isfinite(cfg[key]) or not low<=cfg[key]<=high:
            raise MossTtsError("config")
    if cfg["target_duration_sec"] and not .08 <= cfg["target_duration_sec"] <= cfg["max_audio_sec"]:
        raise MossTtsError("config")
    return cfg


def json_value(raw, limit):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise MossTtsError("input")
            out[k]=v
        return out
    if not isinstance(raw,str): return raw
    try:
        if len(raw.encode("utf-8"))>limit: raise MossTtsError("input")
        return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(MossTtsError("input")))
    except (ValueError,UnicodeError,RecursionError):
        raise MossTtsError("input") from None


def interrupt(raw):
    value=json_value(raw,1024)
    if not isinstance(value,Mapping) or dict(value)!={"action":"interrupt"}:
        raise MossTtsError("input")
    return {"action":"interrupt"}


def request(raw, cfg):
    correlation=uuid4().hex
    # Plain strings remain speech, even if they look like commands. Correlated
    # JSON must arrive as a mapping, or be decoded according to its input MIME.
    if isinstance(raw,Mapping):
        if set(raw)!={"text","request_id"}: raise MossTtsError("input")
        correlation=raw["request_id"];raw=raw["text"]
        if not isinstance(correlation,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}",correlation):
            raise MossTtsError("input")
    try:
        if not isinstance(raw,str) or not raw.strip() or len(raw)>cfg["max_text_chars"] or len(raw.encode("utf-8"))>8192 or "\0" in raw:
            raise MossTtsError("input")
    except UnicodeError:
        raise MossTtsError("input") from None
    return {"text":raw.strip(),"request_id":correlation}


def resolve_path(root, value):
    path=Path(value).expanduser()
    return (Path(root)/path).absolute() if not path.is_absolute() else path


@dataclass
class PreparedAudio:
    owner: object
    path: Path
    output_directory: Path
    metadata: dict
    deadline: float

    def close(self):
        self.owner.cleanup()


def audio_info(path, cfg, raw):
    try:
        info=json_value(raw.decode("utf-8") if isinstance(raw,bytes) else raw,8192)
        if set(info)!={"complete","eos","tokens","segments","sample_rate","frames","clipped_samples"} or info["complete"] is not True or info["eos"] is not True:
            raise ValueError()
        if type(info["tokens"]) is not int or not 1 <= info["tokens"] <= cfg["max_new_tokens"]:
            raise ValueError()
        if type(info["frames"]) is not int or not 0<info["frames"]<=24000*cfg["max_audio_sec"] or type(info["sample_rate"]) is not int or info["sample_rate"]!=24000:
            raise ValueError()
        if type(info["segments"]) is not int or not 1<=info["segments"]<=cfg["max_segments"]:
            raise ValueError()
        if type(info["clipped_samples"]) is not int or not 0<=info["clipped_samples"]<=info["frames"]:
            raise ValueError()
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            file=os.fstat(source.fileno())
            if not stat.S_ISREG(file.st_mode) or not 44<=file.st_size<=24000*cfg["max_audio_sec"]*2+128:
                raise ValueError()
            with wave.open(source,"rb") as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate(),wav.getnframes())!=(1,2,24000,info["frames"]):
                    raise ValueError()
                if len(wav.readframes(info["frames"]+1))!=info["frames"]*2:
                    raise ValueError()
        return info
    except (ValueError,TypeError,KeyError,OSError,EOFError,wave.Error):
        raise MossTtsError("audio_invalid") from None


def prepare_reference(cfg, root, job, deadline, cancel):
    """Optional owned reference, bounded after resampling; no implicit truncation."""
    if not cfg["reference_audio"]:
        return None
    path=resolve_path(root,cfg["reference_audio"])
    if not cfg["reference_audio"] or path.suffix.lower() not in {".wav",".ogg",".webm",".flac",".mp3",".m4a",".mp4"}:
        raise MossTtsError("reference")
    snapshot=job/("reference-source"+path.suffix.lower())
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
        with os.fdopen(fd,"rb") as source, snapshot.open("xb") as copy:
            os.chmod(snapshot,0o600)
            before=os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=33554432:
                raise MossTtsError("reference")
            digest=hashlib.sha256();total=0
            while chunk:=source.read(min(65536,33554433-total)):
                check(deadline,cancel);total+=len(chunk)
                if total>33554432: raise MossTtsError("reference")
                copy.write(chunk);digest.update(chunk)
            after=os.fstat(source.fileno())
            if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns) or total!=before.st_size:
                raise MossTtsError("reference")
        maximum=24000*30
        raw=run_process(["ffmpeg","-v","error","-nostdin","-threads","1","-protocol_whitelist","file,pipe",
            "-format_whitelist","wav,ogg,matroska,webm,flac,mp3,mov","-i",str(snapshot),"-map","0:a:0","-vn",
            "-ac","1","-af",f"aresample=24000,atrim=end_sample={maximum+1}","-ar","24000","-f","f32le","pipe:1"],
            directory=job,deadline=deadline,cancel=cancel,limit=(maximum+1)*4)
        samples=array.array("f");samples.frombytes(raw)
        if sys.byteorder!="little": samples.byteswap()
        if not 12000<=len(samples)<=maximum or any(not math.isfinite(x) or abs(x)>1.001 for x in samples):
            raise MossTtsError("reference")
        if max(abs(x) for x in samples)<1e-5: raise MossTtsError("reference")
        pcm=array.array("h",(round(max(-1,min(1,x))*32767) for x in samples))
        if sys.byteorder!="little": pcm.byteswap()
        with (job/"reference.wav").open("xb") as target:
            os.chmod(job/"reference.wav",0o600)
            with wave.open(target,"wb") as wav:
                wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(24000);wav.writeframes(pcm.tobytes())
        return {"sha256":digest.hexdigest(),"bytes":total,"duration_sec":len(samples)/24000}
    except MossTtsError as exc:
        if exc.code in {"timeout","cancelled"}: raise
        raise MossTtsError("reference") from None
    except (OSError,ValueError,wave.Error):
        raise MossTtsError("reference") from None


def generate(item, cfg, root, storage, cancel=lambda:False):
    cfg=configuration(cfg);item=request(item,cfg)
    storage=Path(storage)
    if not storage.is_absolute() or storage.is_symlink() or not storage.is_dir(): raise MossTtsError("storage")
    assets=resolve_path(root,cfg["asset_directory"])
    if not cfg["asset_directory"] or not assets.is_dir() or assets.is_symlink(): raise MossTtsError("assets_missing")
    python=resolve_path(root,cfg["environment_python"]) if cfg["environment_python"] else Path(sys.executable)
    if not python.is_file() or not os.access(python,os.X_OK): raise MossTtsError("dependency")
    target=resolve_path(root,cfg["output_directory"]) if cfg["output_directory"] else storage/"audio"
    if target.is_symlink() or target.exists() and not target.is_dir(): raise MossTtsError("storage")
    started=time.monotonic();deadline=started+cfg["timeout_sec"]
    owner=tempfile.TemporaryDirectory(prefix="moss-job-",dir=storage);job=Path(owner.name)
    try:
        reference=prepare_reference(cfg,root,job,deadline,cancel)
        envelope={"config":cfg,"request":item,"assets":str(assets),"job":str(job)}
        raw=run_process([str(python),"-I","-B",str(Path(__file__).with_name("worker.py"))],directory=job,
            deadline=deadline,cancel=cancel,limit=8192,stdin=json.dumps(envelope,ensure_ascii=True).encode())
        del envelope
        wav=job/"generated.wav";info=audio_info(wav,cfg,raw)
        output=wav
        if cfg["format"]=="ogg_opus":
            output=job/"generated.ogg"
            run_process(["ffmpeg","-v","error","-nostdin","-n","-threads","1","-protocol_whitelist","file,pipe",
                "-format_whitelist","wav","-i",str(wav),"-map_metadata","-1","-ac","1","-ar","48000",
                "-c:a","libopus","-b:a","96000","-f","ogg",str(output)],directory=job,deadline=deadline,cancel=cancel,limit=1024)
        check(deadline,cancel)
        info.update(request_id=item["request_id"],duration_sec=info["frames"]/24000,
            source_frames=info["frames"],
            format=cfg["format"],sample_rate=48000 if cfg["format"]=="ogg_opus" else 24000,channels=1,
            source_sample_rate=24000,elapsed_sec=round(time.monotonic()-started,4),
            model="OpenMOSS-Team/MOSS-TTS-v1.5",model_revision="cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
            codec_revision="3cd226ba2947efa357ef453bcad111b6eafba782",reference=reference,
            settings={k:cfg[k] for k in ("device","precision","language","target_duration_sec","text_temperature","text_top_p","text_top_k","audio_temperature","audio_top_p","audio_top_k","audio_repetition_penalty","max_audio_sec","max_new_tokens","max_segments")})
        del info["frames"]
        return PreparedAudio(owner,output,target,info,deadline)
    except BaseException:
        owner.cleanup()
        raise


def publish(prepared, cancel=lambda:False):
    """Publish after the listener has drained pending interrupts; never overwrite a file."""
    check(prepared.deadline,cancel)
    target=prepared.output_directory
    if target.is_symlink(): raise MossTtsError("storage")
    target.mkdir(mode=0o700,parents=True,exist_ok=True)
    suffix=".ogg" if prepared.metadata["format"]=="ogg_opus" else ".wav"
    final=target/("moss-"+uuid4().hex+suffix)
    temporary=None
    try:
        fd=os.open(prepared.path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        with os.fdopen(fd,"rb") as source:
            size=os.fstat(source.fileno())
            if not stat.S_ISREG(size.st_mode) or not 0<size.st_size<=33554432: raise MossTtsError("audio_invalid")
            descriptor,temporary=tempfile.mkstemp(prefix=".moss-publish-",dir=target)
            digest=hashlib.sha256();total=0
            with os.fdopen(descriptor,"wb") as output:
                while chunk:=source.read(65536):
                    check(prepared.deadline,cancel);total+=len(chunk)
                    if total>33554432: raise MossTtsError("audio_invalid")
                    output.write(chunk);digest.update(chunk)
                output.flush();os.fsync(output.fileno())
            if total!=size.st_size: raise MossTtsError("audio_invalid")
        check(prepared.deadline,cancel)
        os.link(temporary,final,follow_symlinks=False)
        metadata={**prepared.metadata,"path":str(final),"bytes":total,"sha256":digest.hexdigest(),
            "content_type":"audio/ogg" if suffix==".ogg" else "audio/wav"}
        return metadata
    except OSError:
        raise MossTtsError("storage") from None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
