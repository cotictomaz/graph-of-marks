import os
import gc
import base64
import ollama
import vllm
from typing import Optional, Tuple, Union, Dict, Any


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


def parse_model_entry(entry: Union[str, Dict[str, Any]]) -> Tuple[str, bool]:
    """
    Normalize one ``models:`` list entry into ``(model_name, quantize_fp8)``.

    An entry may be either:
      * a plain string  -> ``"repo/id"``                       (bf16, no quantization)
      * a mapping       -> ``{name: "repo/id", fp8: true}``    (load with FP8 quantization)

    The mapping form lets a config decide FP8 *per model* (e.g. to make a ~12B
    model fit on a 24GB card while leaving smaller models in bf16). ``model`` is
    accepted as an alias for ``name``, and ``quantize_fp8`` for ``fp8``.
    """
    if isinstance(entry, str):
        return entry, False
    if isinstance(entry, dict):
        name = entry.get("name") or entry.get("model")
        if not name:
            raise ValueError(f"Model entry {entry!r} is missing a 'name' (or 'model') key.")
        fp8 = bool(entry.get("fp8", entry.get("quantize_fp8", False)))
        return name, fp8
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
        llm_kwargs: Dict[str, Any] = {"model": self.model_name}
        if quantize_fp8:
            llm_kwargs["quantization"] = "fp8"

        print(f"[VllmVLM] Loading model: {self.model_name}" + (" (fp8)" if quantize_fp8 else ""))
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