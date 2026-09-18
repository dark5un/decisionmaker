"""Torch-gated tests for the decision head/model. Skipped if torch is absent
(the pure engine gates do not need it)."""
import importlib

import pytest

torch = pytest.importorskip("torch")

# The model module imports torch, so it can only be imported once torch is known
# to be present; importlib avoids an import-after-statement (flake8 E402).
model = importlib.import_module("server.src.model")


def test_decision_head_shape():
    head = model.DecisionHead(hidden_size=32)
    x = torch.randn(5, 32)
    out = head(x)
    assert out.shape == (5, 1)


def test_decision_model_forward_with_fake_backbone():
    class FakeBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            h = torch.nn.Linear(8, 16)
            h.weight.data = torch.eye(16, 8)
            h.bias.data.zero_()
            self.lin = h
            self.config = type("C", (), {"hidden_size": 16})()

        def forward(self, input_ids, attention_mask=None, use_cache=None):
            # trivial "hidden state": embeds each token index as one-hot of dim 8
            emb = torch.nn.functional.one_hot(input_ids, num_classes=8).float()  # (P,W,8)
            hidden = self.lin(emb)  # (P,W,16)
            return type("O", (), {"last_hidden_state": hidden})()

    bb = FakeBackbone()
    dm = model.DecisionModel(bb, set_head="none")
    tokens = torch.tensor([[2, 0, 0], [1, 3, 0]], dtype=torch.long)
    attn = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.long)
    lengths = torch.tensor([2, 3], dtype=torch.long)
    scalars = dm(tokens, attn, lengths)
    assert scalars.shape == (2,)
    assert torch.isfinite(scalars).all()
    # deterministic given the input
    s2 = dm(tokens, attn, lengths)
    assert torch.equal(scalars, s2)


def test_head_attention_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        model.DecisionHead(hidden_size=16, set_head="attention")