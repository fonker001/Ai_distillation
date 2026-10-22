"""Top-k distillation with renormalized teacher AND student + auxiliary CE.

Approximation note (not mathematically equal to full-vocabulary KL):
  Full KL is  sum_v q_v * (log q_v - log p_v) over V=32000 tokens.
  We approximate by keeping only the teacher's top-k tokens and
  renormalizing BOTH distributions over that same k-token support:
      q_k = softmax(teacher_logits[topk_ids] / T)
      p_k = softmax(student_logits[topk_ids] / T)
      KD ≈ T^2 * KL(q_k || p_k)
  Mass outside the top-k is ignored by the KD term. The auxiliary
  cross-entropy (kd_alpha < 1) supplies a hard-target signal that
  keeps the student from leaking probability onto tokens that the
  teacher did not rank highly.
"""
import torch
import torch.nn.functional as F


def distillation_loss(student_logits, targets, topk_logits, topk_ids,
                      temperature=2.0, kd_alpha=0.7, ignore_index=-100):
    """
    student_logits: (B, T, V) — predicted logits
    targets:        (B, T)    — hard next-token ids; -100 for pad
    topk_logits:    (B, T, K) — teacher's top-k logits
    topk_ids:       (B, T, K) — teacher's top-k token ids
    """
    B, T, V = student_logits.shape
    valid = (targets != ignore_index)             # (B, T)

    # Teacher distribution, renormalized over the K top-k tokens
    t_logp = F.log_softmax(topk_logits / temperature, dim=-1)   # (B,T,K)
    t_p    = t_logp.exp()

    # Student distribution over the SAME top-k ids, also renormalized over K
    s_logits_k = torch.gather(student_logits, -1, topk_ids)     # (B,T,K)
    s_logp_k   = F.log_softmax(s_logits_k / temperature, dim=-1)

    # KL(t || s), summed over k, averaged over valid positions
    kd = (t_p * (t_logp - s_logp_k)).sum(-1)                    # (B,T)
    kd = kd.masked_select(valid).mean() * (temperature ** 2)

    if kd_alpha >= 1.0:
        return kd
    ce = F.cross_entropy(
        student_logits.reshape(-1, V),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    return kd_alpha * kd + (1.0 - kd_alpha) * ce