"""XLA replacements for two ops inside HF generate; both return exactly what
the original functions return and fall back to them for non-XLA tensors."""
import math
import torch
from transformers import pytorch_utils
from transformers.generation import logits_process, stopping_criteria
from transformers.generation import utils as generation_utils

_original_isin = pytorch_utils.isin_mps_friendly
_original_top_p_call = logits_process.TopPLogitsWarper.__call__
_original_min_length_call = logits_process.MinLengthLogitsProcessor.__call__
_patched = False


def _isin_xla(elements, test_elements):
    # torch.isin has no XLA lowering (CPU fallback on every decoding step)
    if elements.device.type != "xla":
        return _original_isin(elements, test_elements)
    test_elements = torch.as_tensor(
        test_elements, device=elements.device).flatten()
    return (elements.unsqueeze(-1) == test_elements).any(-1)


def _top_p_call_xla(self, input_ids, scores):
    # same as TopPLogitsWarper.__call__, except the scatter back to vocabulary
    # order (a serial 50k-index loop on the TPU, ~200 ms per call)
    if scores.device.type != "xla":
        return _original_top_p_call(self, input_ids, scores)
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
    sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0
    # sorted_indices is a permutation: sorting 2 * index + flag puts the
    # flags back in vocabulary order
    keys = sorted_indices.int() * 2 + sorted_indices_to_remove.int()
    indices_to_remove = torch.sort(keys, dim=-1)[0].remainder(2).bool()
    return scores.masked_fill(indices_to_remove, self.filter_value)


def _min_length_call_xla(self, input_ids, scores):
    # same as MinLengthLogitsProcessor.__call__, but the EOS mask over the
    # vocabulary is built once per processor instead of on every step
    if scores.device.type != "xla":
        return _original_min_length_call(self, input_ids, scores)
    if getattr(self, "_xla_eos_mask", None) is None:
        vocab_tensor = torch.arange(scores.shape[-1], device=scores.device)
        self._xla_eos_mask = _isin_xla(vocab_tensor, self.eos_token_id)
    scores_processed = scores.clone()
    if input_ids.shape[-1] < self.min_length:
        scores_processed = torch.where(self._xla_eos_mask, -math.inf, scores)
    return scores_processed


def patch_hf_generation_for_xla():
    global _patched
    if _patched:
        return
    for module in (logits_process, stopping_criteria, generation_utils):
        module.isin_mps_friendly = _isin_xla
    logits_process.TopPLogitsWarper.__call__ = _top_p_call_xla
    logits_process.MinLengthLogitsProcessor.__call__ = _min_length_call_xla
    _patched = True
