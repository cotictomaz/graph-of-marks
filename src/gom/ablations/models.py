import os
import base64
import ollama
import vllm
from typing import Optional, Tuple, Union, Dict, Any


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

    def __init__(self, model_name: str = "qwen2.5vl:3b", system_prompt: str = ""):
        self.model_name = model_name
        self.system_prompt = system_prompt
        print(f"[OllamaVLM] Model initialized: {self.model_name}")

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
                options={"num_ctx": 8192},
            )
            # Support both attribute-style (ollama >= 0.2) and dict-style access
            msg = response.message if hasattr(response, "message") else response["message"]
            return msg.content if hasattr(msg, "content") else msg["content"]
        except Exception as e:
            print(f"[OllamaVLM Error] Inference failed: {e}")
            return f"Error during inference {e}"


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
    ):
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:
            raise ImportError("vllm is not installed. Install it with: pip install vllm") from e

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.quantize_fp8 = quantize_fp8

        # Build the LLM kwargs; only add `quantization` when FP8 is requested so
        # bf16 loads (the default) behave exactly as before. On-the-fly FP8
        # roughly halves the weight footprint (e.g. a ~12B model ~24GB -> ~13GB),
        # letting it fit on a single 24GB GPU with negligible accuracy loss.
        llm_kwargs: Dict[str, Any] = {"model": self.model_name}
        if quantize_fp8:
            llm_kwargs["quantization"] = "fp8"

        print(f"[VllmVLM] Loading model: {self.model_name}" + (" (fp8)" if quantize_fp8 else ""))
        self.llm = LLM(**llm_kwargs)
        self.sampling_params = SamplingParams(max_tokens=512, temperature=0.0)
        print(f"[VllmVLM] Model loaded: {self.model_name}")

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