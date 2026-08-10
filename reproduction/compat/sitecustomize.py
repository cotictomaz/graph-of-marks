"""Force the vision-tower attention backend when GOM_VIT_ATTN_BACKEND is set.

Qwen2.5-VL cannot start on this Blackwell GPU (sm_120) with either backend vLLM
will choose on its own:

* `_Backend.XFORMERS` (the fallback) dispatches xformers' vendored
  FlashAttention-3 *Hopper* kernel -> "CUDA error ... invalid argument".
* `_Backend.FLASH_ATTN` needs the `flash_attn` package; routing it to vLLM's own
  build (see `flash_attn.py`) fails because that build rejects the ViT's
  head_dim of 80 ("headdim not being a multiple of 32").

`_Backend.TORCH_SDPA` works and Qwen2.5-VL supports it, but the only public
selector, `VLLM_ATTENTION_BACKEND`, is shared with the language model, where
CUDA rejects TORCH_SDPA outright. So the vision tower has to be steered on its
own. `vision.get_vit_attn_backend` delegates to `current_platform` at call time,
so patching the platform method is enough - and Python imports `sitecustomize`
in every interpreter, which is how this reaches vLLM's spawned EngineCore
process as well as the parent.

Set GOM_VIT_ATTN_BACKEND=TORCH_SDPA to enable. Unset, this file does nothing.
"""
import builtins
import os

_TARGET = os.environ.get("GOM_VIT_ATTN_BACKEND")

if _TARGET:
    _real_import = builtins.__import__

    def _patch_platform() -> bool:
        try:
            from vllm.platforms import _Backend
            from vllm.platforms.cuda import CudaPlatformBase
        except Exception:
            return False
        backend = getattr(_Backend, _TARGET, None)
        if backend is None:
            return False
        if getattr(CudaPlatformBase, "_gom_vit_patched", False):
            return True
        CudaPlatformBase.get_vit_attn_backend = classmethod(
            lambda cls, support_fa=False: backend
        )
        CudaPlatformBase._gom_vit_patched = True
        print(f"[gom] vision-tower attention backend forced to {_TARGET}", flush=True)
        return True

    def _import(name, *args, **kwargs):
        module = _real_import(name, *args, **kwargs)
        if name.startswith("vllm.platforms") and not getattr(
            builtins, "_gom_vit_done", False
        ):
            if _patch_platform():
                builtins._gom_vit_done = True
        return module

    builtins.__import__ = _import
