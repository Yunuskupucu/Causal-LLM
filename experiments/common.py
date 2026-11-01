"""Deneyler arasında paylaşılan yükleme yardımcıları."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data import load_corpus, make_splits   # noqa: E402
from src.model import MiniGPT                   # noqa: E402
from src.tokenizer import CharTokenizer         # noqa: E402
from src.utils import RES_DIR                   # noqa: E402


def load_everything():
    """Külliyat, tokenizer, bölünmüş veri ve eğitilmiş modeli döndürür."""
    text = load_corpus()
    tok = CharTokenizer(text)
    train_ids, val_ids = make_splits(text, tok)
    ck = torch.load(RES_DIR / "model.pt", map_location="cpu")
    cfg = ck["config"]
    model = MiniGPT(tok.vocab_size, cfg["BLOCK_SIZE"], cfg["N_LAYER"],
                    cfg["N_HEAD"], cfg["D_MODEL"], cfg["DROPOUT"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return text, tok, train_ids, val_ids, model


@torch.no_grad()
def next_token_logits(model, ids: torch.Tensor) -> torch.Tensor:
    """Bağlam x_{<=t} için sıradaki belirtecin logit vektörü z (|V|,)."""
    ids = ids[-model.block_size:].unsqueeze(0)
    logits, _ = model(ids)
    return logits[0, -1]
