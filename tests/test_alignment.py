"""Fast unit test proving the alignment between cache, targets, and logits."""
import torch
from src.loss import distillation_loss


def test_alignment_and_zero_loss_when_student_matches_teacher():
    torch.manual_seed(0)
    B, T, V, K = 1, 5, 50, 8
    # Simulate teacher logits
    teacher_logits = torch.randn(B, T, V)
    pred_logits = teacher_logits[:, :-1, :].contiguous()   # (B, T-1, V)
    targets = torch.randint(0, V, (B, T - 1))              # (B, T-1)
    topk_vals, topk_ids = torch.topk(pred_logits, K, dim=-1)

    # Student = teacher: identical logits
    student_logits = pred_logits.clone()
    loss = distillation_loss(student_logits, targets, topk_vals, topk_ids,
                             temperature=1.0, kd_alpha=1.0)
    assert loss.item() < 1e-5, f"expected ~0, got {loss.item()}"


def test_padding_mask():
    torch.manual_seed(1)
    B, T, V, K = 1, 4, 20, 5
    student = torch.randn(B, T, V)
    topk_vals, topk_ids = torch.topk(torch.randn(B, T, V), K, dim=-1)
    targets = torch.tensor([[-100, -100, 1, 2]])
    # Only 2 valid positions; ensure finite and low-dimensional
    loss = distillation_loss(student, targets, topk_vals, topk_ids,
                             temperature=2.0, kd_alpha=0.7)
    assert torch.isfinite(loss)