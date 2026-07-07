import os
import gc
import base64
import ollama
import vllm
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any


# Generation budgets. Reasoning / "thinking" models spend a large number of
# tokens on their <think> trace BEFORE emitting the final answer, so a 512-token
# cap (fine for a terse VQA answer) truncates them mid-reasoning and the answer
# never appears — which ablations/evaluation.py would then score as empty.
# We therefore auto-raise the cap for those models and keep the cheap default
# for ordinary ones. `answer extraction` still strips the trace at eval time.
DEFAULT_MAX_TOKENS = 512
REASONING_MAX_TOKENS = 2048

# Substrings (case-insensitive) that mark a model as reasoning/thinking. Kept
# broad so it also catches families we might add later (o1, R1, QvQ, ...).
_REASONING_NAME_HINTS = (
    "thinking", "reasoning", "-o1", "o1-", "llamav-o1",
    "qvq", "-r1", "r1-", "deepseek-r1", "cot",
)


def is_reasoning_model(model_name: str) -> bool:
    """Heuristic: does this model emit a reasoning trace before its answer?"""
    n = (model_name or "").lower()
    return any(hint in n for hint in _REASONING_NAME_HINTS)


def resolve_max_tokens(model_name: str, explicit: Optional[int] = None) -> int:
    """Pick the generation cap: explicit override wins, else auto by model type."""
    if explicit is not None:
        return explicit
    return REASONING_MAX_TOKENS if is_reasoning_model(model_name) else DEFAULT_MAX_TOKENS


@dataclass
class ModelSpec:
    """Normalized description of one ``models:`` list entry.

    Beyond the model name and FP8 flag, an entry may carry **per-model load
    overrides** so a single config can size each model independently — a large
    model can get a smaller ``max_model_len`` / a vision-token cap while smaller
    models keep the defaults. All overrides default to ``None`` ("use the
    ``VllmVLM`` default").

    * ``max_model_len`` — context window cap for *this* model. Size it to the
      model's **measured** max GoM-prompt length; the hard-coded default is only
      a fallback and is the classic cause of "decoder prompt ... longer than the
      maximum model length" when it is too small.
    * ``max_pixels`` — Qwen-style vision-token cap (fewer pixels -> fewer image
      tokens -> shorter prompt). The real lever for oversized image prompts
      (e.g. Qwen3-VL's ~16k-token GoM prompts) — cheaper on KV than inflating
      ``max_model_len``.
    * ``max_tokens`` — generation cap override (else auto by reasoning-model
      detection).
    """
    name: str
    quantize_fp8: bool = False
    max_model_len: Optional[int] = None
    max_pixels: Optional[int] = None
    max_tokens: Optional[int] = None


def parse_model_entry(entry: Union[str, Dict[str, Any]]) -> ModelSpec:
    """
    Normalize one ``models:`` list entry into a :class:`ModelSpec`.

    An entry may be either:
      * a plain string  -> ``"repo/id"``                    (bf16, all defaults)
      * a mapping       -> ``{name: "repo/id", fp8: true,
                              max_model_len: 8192, max_pixels: 262144}``

    The mapping form lets a config decide FP8 and load sizing *per model* (e.g.
    to make a ~12B model fit on a 24GB card, or cap a model's vision tokens so
    its GoM prompt fits ``max_model_len``). ``model`` is accepted as an alias for
    ``name`` and ``quantize_fp8`` for ``fp8``.
    """
    if isinstance(entry, str):
        return ModelSpec(name=entry)
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("model")
        if not name:
            raise ValueError(f"Model entry {entry!r} is missing a 'name' (or 'model') key.")
        fp8 = bool(entry.get("fp8", entry.get("quantize_fp8", False)))
        return ModelSpec(
            name=name,
            quantize_fp8=fp8,
            max_model_len=entry.get("max_model_len"),
            max_pixels=entry.get("max_pixels"),
            max_tokens=entry.get("max_tokens"),
        )
    raise ValueError(
        f"Unsupported model entry {entry!r}: expected a string or a mapping with a 'name' key."
    )


class OllamaVLM:
    """
    Wrapper for Vision-Language models served by a local Ollama server.
    Supports models like Qwen, Llama-Vision, and Moondream.
    """

    def __init__(self, model_name: str = "qwen2.5vl:3b", system_prompt: str = "", max_tokens: Optional[int] = None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = resolve_max_tokens(model_name, max_tokens)
        # Reasoning models need a wider context to hold the <think> trace + answer.
        self.num_ctx = 16384 if is_reasoning_model(model_name) else 8192
        print(f"[OllamaVLM] Model initialized: {self.model_name} "
              f"(num_predict={self.max_tokens}, num_ctx={self.num_ctx})")

    def generate(self, prompt: str, image_path: str) -> str:
        if not os.path.exists(image_path):
            print(f"[OllamaVLM Error] Image not found: {image_path}")
            return "Error: Image not found"

        try:
            with open(image_path, "rb") as f:
                img_bytes = f.read()
        except Exception as e:
            print(f"[OllamaVLM Error] Cannot read image: {e}")
            return "Error reading image"

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt, "images": [img_bytes]})

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={"num_ctx": self.num_ctx, "num_predict": self.max_tokens},
            )
            # Support both attribute-style (ollama >= 0.2) and dict-style access
            msg = response.message if hasattr(response, "message") else response["message"]
            return msg.content if hasattr(msg, "content") else msg["content"]
        except Exception as e:
            print(f"[OllamaVLM Error] Inference failed: {e}")
            return f"Error during inference {e}"

    def shutdown(self) -> None:
        """Ask the Ollama daemon to unload this model from VRAM.

        Ollama serves models from its own server process, so this wrapper holds
        no GPU memory itself — there is nothing to free in-process. We simply
        nudge the daemon to evict the model now (``keep_alive=0``) instead of
        waiting out its keep-alive window, so the next model starts on a clean
        GPU. Best-effort: a failure here is harmless.
        """
        try:
            ollama.generate(model=self.model_name, prompt="", keep_alive=0)
        except Exception as e:
            print(f"[OllamaVLM] unload warning: {e}")


class VllmVLM:
    """
    Wrapper for Vision-Language models served by vllm.
    The model is loaded into GPU memory at __init__ time (expensive once,
    then fast for repeated generate() calls).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        system_prompt: str = "",
        quantize_fp8: bool = False,
        max_tokens: Optional[int] = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = 8192,
        max_num_batched_tokens: Optional[int] = None,
        enforce_eager: bool = True,
        max_num_seqs: Optional[int] = 8,
        trust_remote_code: bool = True,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        limit_mm_per_prompt: Optional[Dict[str, int]] = None,
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError("vllm is not installed. Install it with: pip install vllm") from e

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.quantize_fp8 = quantize_fp8
        # Auto-raise the token budget for reasoning models unless overridden.
        self.max_tokens = resolve_max_tokens(model_name, max_tokens)

        # Build the LLM kwargs; only add `quantization` when FP8 is requested so
        # bf16 loads (the default) behave exactly as before. On-the-fly FP8
        # roughly halves the weight footprint (e.g. a ~12B model ~24GB -> ~13GB),
        # letting it fit on a single 24GB GPU with negligible accuracy loss.
        # (NB: FP8 is emulated via Marlin on Ampere/sm_86 and fails on some dims —
        # use bf16 on the RTX 3090; see CLAUDE.md.)
        llm_kwargs: Dict[str, Any] = {"model": self.model_name}
        if quantize_fp8:
            llm_kwargs["quantization"] = "fp8"

        # Some VLMs ship their processor/model as custom code in the HF repo
        # (e.g. OpenGVLab/InternVL3_5-8B) and vLLM refuses to load them without
        # this flag ("contains custom code which must be executed..."). Harmless
        # for models with a native vLLM implementation (e.g. Qwen3-VL).
        if trust_remote_code:
            llm_kwargs["trust_remote_code"] = True

        # --- Context length (KV block size + the HARD prompt-length ceiling) ---
        # `max_model_len` is BOTH the per-sequence KV size vLLM reserves AND the
        # ceiling a prompt may not exceed. The native windows these VLMs advertise
        # are absurd for VQA (Qwen3-VL = 262144) and would size the KV blocks for
        # that, so we always cap it. This is the single most failure-prone knob: a
        # GoM prompt is dominated by the image's *vision* tokens, and those counts
        # differ ~4x between encoders (InternVL ~3.5-4.3k, Qwen3-VL ~16k for the
        # SAME scene). If `max_model_len` is below a model's actual prompt length,
        # EVERY request errors with "decoder prompt ... longer than the maximum
        # model length". So size it to the model's MEASURED max prompt length
        # (per-model, via the config entry's `max_model_len:` key); the 8192
        # default is only a fallback. For a model whose prompts are still too long
        # (Qwen3-VL), cap the vision tokens with `max_pixels` below rather than
        # inflating `max_model_len` — a huge window blows the KV budget and does
        # NOT shorten the prompt. Pass None to use the model's native maximum.
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        # --- Startup profiling batch (MUST be able to hold one whole image) ----
        # vLLM's startup `profile_run` allocates transient activation for
        # `max_num_batched_tokens` tokens on top of the weights (NOT capped by
        # `gpu_memory_utilization`), so an over-large value can OOM at profiling.
        # But a too-SMALL value is a multimodal footgun: a single image's vision
        # tokens can't be split across prefill chunks, so if this is below one
        # image's token count the engine fails with "multimodal item cannot fit
        # into max_num_batched_tokens". We therefore default it to `max_model_len`
        # (guaranteeing any single prompt+image fits one chunk); override only if
        # profiling OOMs and you know a single image fits the smaller value.
        if max_num_batched_tokens is None and max_model_len is not None:
            max_num_batched_tokens = max_model_len
        if max_num_batched_tokens is not None:
            llm_kwargs["max_num_batched_tokens"] = max_num_batched_tokens

        # Skip CUDA-graph capture. After profiling, vLLM otherwise captures ~70
        # batch sizes into CUDA graphs, each reserving extra VRAM — enough to tip
        # a near-full 24GB card into OOM. Eager mode is slightly slower per step
        # but removes that reservation; negligible for these small VQA runs.
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True

        # Cap the max concurrent sequences. vLLM warms up its sampler with
        # `max_num_seqs` (default 256) dummy requests, allocating logits/sampling
        # tensors for all of them at once — which OOMs a model that already needs
        # most of a 24GB card. VQA inference here is batch_size=1, so a small pool
        # (8) is plenty and keeps the warmup + scheduler memory tiny.
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = max_num_seqs

        # Single-image VQA: profile for at most one image per prompt (vLLM's
        # default assumes more, inflating the profiling activation).
        if limit_mm_per_prompt is None:
            limit_mm_per_prompt = {"image": 1}
        llm_kwargs["limit_mm_per_prompt"] = limit_mm_per_prompt

        # --- Vision-token cap (the real lever for oversized image prompts) -----
        # For Qwen-family processors, `max_pixels`/`min_pixels` bound how many
        # visual tokens an image expands to (fewer pixels -> fewer tokens ->
        # shorter prompt). This is how a ~16k-token Qwen3-VL GoM prompt is brought
        # under a sane `max_model_len` WITHOUT touching the KV budget. Threaded
        # through `mm_processor_kwargs`; a caller may also pass an explicit
        # `mm_processor_kwargs` (e.g. InternVL's `max_dynamic_patch`), whose keys
        # win over the pixel shortcuts.
        mm_kwargs: Dict[str, Any] = dict(mm_processor_kwargs or {})
        if max_pixels is not None:
            mm_kwargs.setdefault("max_pixels", max_pixels)
        if min_pixels is not None:
            mm_kwargs.setdefault("min_pixels", min_pixels)
        if mm_kwargs:
            llm_kwargs["mm_processor_kwargs"] = mm_kwargs

        # --- GPU memory budget -------------------------------------------------
        # `gpu_memory_utilization` is the TOTAL-budget cap, NOT "free memory
        # demanded at startup": vLLM sizes the KV cache as
        #   KV = gpu_memory_utilization * total - weights - activation - overhead
        # so a HIGHER value buys MORE KV (an 8B model is ~16GB of weights, not the
        # ~22GB an earlier comment claimed — the rest of a full card is reserved,
        # controllable KV). `main.py` frees the GoM preprocessor
        # (release_preprocessor) BEFORE the VLM loads, so the card is clean and
        # vLLM's normal 0.90 default just works. (The old auto-sizer that read
        # free VRAM and crept to ~0.96 has been removed: it duplicated the
        # preprocessor-release fix and drifted into a fragile startup-fail band.)
        # Raise this toward ~0.95 only to buy extra KV once the weights already fit.
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

        print(
            f"[VllmVLM] Loading model: {self.model_name}"
            + (" (fp8)" if quantize_fp8 else "")
            + (f" (gpu_mem_util={gpu_memory_utilization:.2f})" if gpu_memory_utilization else "")
            + (f" (max_model_len={max_model_len})" if max_model_len else "")
            + (f" (mm={mm_kwargs})" if mm_kwargs else "")
        )
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(max_tokens=self.max_tokens, temperature=0.0)
        print(f"[VllmVLM] Model loaded: {self.model_name} (max_tokens={self.max_tokens})")

    def generate(self, prompt: str, image_path: str) -> str:
        if not os.path.exists(image_path):
            print(f"[VllmVLM Error] Image not found: {image_path}")
            return "Error: Image not found"

        try:
            with open(image_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"[VllmVLM Error] Cannot read image: {e}")
            return "Error reading image"

        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"

        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        })

        try:
            outputs = self.llm.chat(messages, sampling_params=self.sampling_params)
            return outputs[0].outputs[0].text
        except Exception as e:
            print(f"[VllmVLM Error] Inference failed: {e}")
            return f"Error during inference {e}"

    def shutdown(self) -> None:
        """Explicitly tear down this vLLM engine and release its GPU memory.

        vLLM does not reliably free VRAM when the Python object is dropped: the
        CUDA context, the KV-cache blocks and (multi-GPU) NCCL state outlive
        garbage collection and persist until the process exits. When several
        models run back-to-back in one job this leaves the previous model's
        weights resident while the next model loads, and can OOM. We therefore
        tear the engine down by hand before moving on.

        The engine-internal layout differs across vLLM versions (V0 exposes
        ``llm_engine.model_executor``, V1 ``llm_engine.engine_core``), so every
        step is best-effort and the method is idempotent — the actual cache
        flush (``gc`` + ``empty_cache``) is done by ``release_model`` right
        after this returns.
        """
        if getattr(self, "_is_shutdown", False):
            return
        self._is_shutdown = True

        # Drop the engine's references to the model executor/workers so the
        # weight tensors lose their last owner before the collect in release_model.
        llm = getattr(self, "llm", None)
        if llm is not None:
            for attr_path in ("llm_engine.model_executor", "llm_engine.engine_core"):
                obj = llm
                try:
                    *parents, last = attr_path.split(".")
                    for p in parents:
                        obj = getattr(obj, p)
                    if hasattr(obj, last):
                        delattr(obj, last)
                except Exception:
                    pass
        self.llm = None

        # Tear down vLLM's parallel/distributed state if this build exposes it.
        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
                destroy_distributed_environment,
            )
            destroy_model_parallel()
            destroy_distributed_environment()
        except Exception:
            pass


def release_model(model: Any) -> None:
    """Free any GPU memory held by a VLM wrapper before the next one loads.

    Calls the wrapper's own ``shutdown()`` (vLLM tears its engine down; Ollama
    asks its daemon to unload), then forces a garbage collection and a CUDA
    cache flush — in that order, so the cache flush actually reclaims the blocks
    the engine just released. Entirely best-effort and exception-safe: cleanup
    must never abort an experiment run, so callers can invoke it from a
    ``finally`` without guarding it.
    """
    if model is None:
        return
    try:
        shutdown = getattr(model, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as e:
        print(f"[release_model] shutdown warning: {e}")

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass