import torch

QUANT_VALUE_BYTES = {"none": 4, "fp16": 2, "int8": 1}
COMPRESSION_INDEX_BYTES = 4

def quantize_values(flat_values, mode):
    """Round-trips values through the target precision so the returned tensor carries
    real quantization error, not just an assumed byte count."""
    if mode == "none":
        return flat_values
    if mode == "fp16":
        return flat_values.half().float()
    if mode == "int8":
        absmax = flat_values.abs().max()
        if absmax < 1e-12:
            return torch.zeros_like(flat_values)
        scale = absmax / 127.0
        q_int = torch.clamp(torch.round(flat_values / scale), -127, 127)
        return q_int * scale
    raise ValueError(f"Unknown quant mode: {mode}")

def compress_delta(delta, topk_fraction=1.0, quant_mode="none"):
    """Top-k sparsifies each tensor (if topk_fraction < 1.0), then quantizes the retained
    values (if quant_mode != 'none'). Both steps genuinely modify the returned values, so
    the result carries the same numerical degradation a real compressed transmission
    would introduce.
    """
    value_bytes = QUANT_VALUE_BYTES[quant_mode]
    compressed = {}
    total_bytes = 0
    for k, v in delta.items():
        flat = v.flatten()
        if topk_fraction >= 1.0:
            mask = torch.ones_like(flat, dtype=torch.bool)
            index_bytes = 0
        else:
            k_keep = max(1, int(flat.numel() * topk_fraction))
            threshold = torch.topk(flat.abs(), k_keep, largest=True).values.min()
            mask = flat.abs() >= threshold
            index_bytes = COMPRESSION_INDEX_BYTES

        nnz = int(mask.sum().item())
        kept_vals = torch.where(mask, flat, torch.zeros_like(flat))
        kept_vals = quantize_values(kept_vals, quant_mode)

        total_bytes += nnz * (index_bytes + value_bytes)
        if quant_mode == "int8" and nnz > 0:
            total_bytes += 4  # one fp32 scale factor per tensor, shared by all its retained elements
        compressed[k] = kept_vals.view_as(v)
    return compressed, total_bytes