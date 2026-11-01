"""Torch tests. Skipped automatically when torch is not installed."""
import pytest

torch = pytest.importorskip("torch")

from signbridge_brain.models.bilstm_ctc import BiLSTMCTC  # noqa: E402
from signbridge_brain.models.fallback import WindowClassifier, sliding_window_decode  # noqa: E402


def test_forward_shapes_and_normalised():
    m = BiLSTMCTC(input_dim=20, num_classes=7, proj_dim=16, hidden_size=8)
    x = torch.randn(3, 30, 20)
    lens = torch.tensor([30, 25, 10])
    logp, out_lens = m(x, lens)
    assert logp.shape == (3, 30, 7)
    assert torch.allclose(logp.exp().sum(-1), torch.ones(3, 30), atol=1e-4)
    assert out_lens.tolist() == [30, 25, 10]


def test_ctc_loss_finite_and_backprop():
    m = BiLSTMCTC(input_dim=20, num_classes=7, proj_dim=16, hidden_size=8)
    x = torch.randn(2, 40, 20)
    lens = torch.tensor([40, 32])
    y = torch.tensor([[1, 2, 3], [4, 5, 0]])
    yl = torch.tensor([3, 2])
    logp, ol = m(x, lens)
    loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)(logp.transpose(0, 1), y, ol, yl)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None for p in m.parameters())


def test_ctc_overfits_tiny_batch():
    torch.manual_seed(0)
    m = BiLSTMCTC(input_dim=8, num_classes=4, proj_dim=32, hidden_size=32, num_layers=1, dropout=0.0,
                  input_dropout=0.0)
    # 2 distinct fixed sequences with different labels
    x = torch.randn(2, 40, 8)
    lens = torch.tensor([40, 40])
    y = torch.tensor([[1, 2], [3, 1]])
    yl = torch.tensor([2, 2])
    crit = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    first = None
    for _ in range(120):
        logp, ol = m(x, lens)
        loss = crit(logp.transpose(0, 1), y, ol, yl)
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = first if first is not None else float(loss)
    assert float(loss) < 0.3 * first


def test_sliding_window_decode_merges_duplicates():
    import numpy as np

    class Fixed(torch.nn.Module):  # always confident "class 2" on every window
        def forward(self, x):
            out = torch.full((x.shape[0], 4), -10.0)
            out[:, 2] = 10.0
            return out

    ids, conf = sliding_window_decode(Fixed(), np.zeros((60, 5), dtype=np.float32), window=30, stride=5)
    assert ids == [2] and conf > 0.99


def test_window_classifier_shape():
    m = WindowClassifier(input_dim=10, num_classes=5)
    assert m(torch.randn(4, 30, 10)).shape == (4, 5)
