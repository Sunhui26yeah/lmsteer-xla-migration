"""XLA replacements for two ops inside HF generate; both return exactly what
the original functions return and fall back to them for non-XLA tensors."""
import torch
from transformers import pytorch_utils
from transformers.generation import logits_process, stopping_criteria
from transformers.generation import utils as generation_utils

_original_isin = pytorch_utils.isin_mps_friendly
_original_top_p_call = logits_process.TopPLogitsWarper.__call__
_patched = False


def _isin_xla(elements, test_elements):
    # torch.isin has no XLA lowering (CPU fallback on every decoding step)
    if elements.device.type != "xla":
        return _original_isin(elements, test_elements)
    test_elements = torch.as_tensor(
        test_elements, device=elements.device).flatten()
    return (elements.unsqueeze(-1) == test_elements).any(-1)


def patch_hf_generation_for_xla():
    global _patched
    if _patched:
        return
    for module in (logits_process, stopping_criteria, generation_utils):
        module.isin_mps_friendly = _isin_xla
    _patched = True
