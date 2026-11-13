"""Versión por lotes (densa) de B6 / P1 / P2 sobre ``GraphTensors``.

La capa GATv2 de la tesis NO se reimplementa: ``DenseThesisGATv2`` contiene una
instancia de ``ThesisGATv2Layer`` (model.gat_variant: thesis, W_r + atención
sobre a_ij + residual + LayerNorm) y usa SUS parámetros, calculando exactamente
la misma función de forma vectorizada sobre un lote de grafos con padding. El
test ``tests/test_batched_models.py`` comprueba que la salida coincide con
``ThesisGATv2Layer.forward`` (atol 1e-5) grafo por grafo. Se hace solo por
velocidad: entrenar 5 semillas x varios sistemas en CPU con bucles por nodo no
cabe en el presupuesto.

Los tres sistemas comparten codificadores, pooling y clasificador:
  B6 = codificadores -> pooling -> MLP           (sin aristas)
  P1 = codificadores -> 2 capas GATv2 (g=1) -> pooling -> MLP
  P2 = P1 con g_j = clip(c q r (1-u), 0, 1) escalando el mensaje del emisor
Pooling: ``mean`` o ``attention`` (agrupamiento por atención de la tesis),
elegido en validación e igual para los tres.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from uafeg.models.p1_graph import EDGE_FEATURE_DIM, SELF_LOOP_RELATION, THESIS_RELATION_INDEX, ThesisGATv2Layer
from uafeg.thesis_graph import F_MAX

NUM_CLASSES = 3
SELF = THESIS_RELATION_INDEX[SELF_LOOP_RELATION]


def collate(graphs, use_edge_attributes: bool = True) -> dict:
    """Lista de GraphTensors -> tensores densos con padding."""
    b = len(graphs)
    n = max(g.n for g in graphs)
    x = torch.zeros(b, n, F_MAX)
    kind = torch.zeros(b, n, dtype=torch.long)
    mask = torch.zeros(b, n, dtype=torch.bool)
    gate = torch.ones(b, n)
    rel = torch.full((b, n, n), -1, dtype=torch.long)       # rel[b, dst, src]
    attr = torch.zeros(b, n, n, EDGE_FEATURE_DIM)
    cov = torch.zeros(b)
    for i, g in enumerate(graphs):
        k = g.n
        x[i, :k] = torch.from_numpy(g.x)
        kind[i, :k] = torch.from_numpy(g.kind)
        mask[i, :k] = True
        c, q, r, u = torch.from_numpy(g.gate).T
        gate[i, :k] = torch.clamp(c * q * r * (1 - u), 0, 1)
        idx = torch.arange(k)
        rel[i, idx, idx] = SELF
        for src, dst, relation, a in g.edges:
            rel[i, dst, src] = relation
            if use_edge_attributes:
                attr[i, dst, src] = torch.tensor(a, dtype=torch.float32)
        cov[i] = g.coverage_bar
    return {"x": x, "kind": kind, "mask": mask, "gate": gate, "rel": rel, "attr": attr, "coverage": cov}


class DenseThesisGATv2(nn.Module):
    """Misma función que ``ThesisGATv2Layer`` (cuyos parámetros contiene), por lotes."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.layer = ThesisGATv2Layer(dim, heads=heads)

    def forward(self, h, rel, attr, gate):
        L = self.layer
        b, n, d = h.shape
        cand = rel >= 0                                             # (B, dst, src)
        score = (L.W_src(h).unsqueeze(1) + L.W_dst(h).unsqueeze(2) + L.W_attr(attr))  # (B,n,n,d)
        score = L.leaky_relu(score).view(b, n, n, L.heads, L.head_dim)
        e = torch.einsum("bijhd,hd->bijh", score, L.attn)
        e = e.masked_fill(~cand.unsqueeze(-1), float("-inf"))
        alpha = torch.softmax(e, dim=2)
        alpha = torch.nan_to_num(alpha, nan=0.0)                    # padded rows have no candidates
        values = torch.stack([w(h) for w in L.relation_value], dim=1)  # (B, R, n_src, d)
        r_idx = rel.clamp(min=0)                                    # (B, dst, src)
        src_idx = torch.arange(n).view(1, 1, n).expand(b, n, n)
        msg = values[torch.arange(b).view(b, 1, 1), r_idx, src_idx]  # (B, dst, src, d)
        msg = msg * gate.view(b, 1, n, 1)
        msg = msg.view(b, n, n, L.heads, L.head_dim)
        agg = (alpha.unsqueeze(-1) * msg).sum(dim=2).reshape(b, n, d)
        return L.norm(h + L.activation(agg))


class EvidenceModel(nn.Module):
    def __init__(self, system: str, n_kinds: int, dim: int = 64, hidden: int = 64, heads: int = 2,
                 layers: int = 2, dropout: float = 0.1, pooling: str = "attention"):
        super().__init__()
        if system not in ("B6", "P1", "P2"):
            raise ValueError(system)
        if pooling not in ("mean", "attention"):
            raise ValueError(pooling)
        self.system, self.pooling = system, pooling
        self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(F_MAX, hidden), nn.ReLU(), nn.Linear(hidden, dim))
                                       for _ in range(n_kinds)])
        self.gat = nn.ModuleList([DenseThesisGATv2(dim, heads) for _ in range(layers)]) if system != "B6" else None
        self.drop = nn.Dropout(dropout)
        self.attn_pool = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.classifier = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, NUM_CLASSES))

    def encode(self, x, kind):
        out = torch.zeros(*x.shape[:2], self.encoders[0][-1].out_features)
        for k, enc in enumerate(self.encoders):
            sel = kind == k
            if sel.any():
                out[sel] = enc(x[sel])
        return out

    def forward(self, batch, gate_override: torch.Tensor | None = None):
        mask = batch["mask"]
        h = self.encode(batch["x"], batch["kind"])
        if self.gat is not None:
            gate = batch["gate"] if self.system == "P2" else torch.ones_like(batch["gate"])
            if gate_override is not None:
                gate = gate_override
            for layer in self.gat:
                h = self.drop(layer(h, batch["rel"], batch["attr"], gate))
        if self.pooling == "mean":
            w = mask.float() / mask.float().sum(1, keepdim=True)
        else:
            s = self.attn_pool(h).squeeze(-1).masked_fill(~mask, float("-inf"))
            w = torch.softmax(s, dim=1)
        pooled = (w.unsqueeze(-1) * h).sum(1)
        return self.classifier(pooled)


# ----------------------------------------------------------------------- flat baselines
class MLP(nn.Module):
    def __init__(self, d_in, hidden=64, dropout=0.1, d_out=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


class DBF(nn.Module):
    """B7: Discounted Belief Fusion (Bezirganyan et al., AISTATS 2025), implementación
    propia a partir del artículo (no hay código oficial en este entorno).

    Cada modalidad m produce evidencia de Dirichlet e_m = softplus(f_m(x_m)) y una
    opinión subjetiva (b_m = e_m / S_m, u_m = K / S_m). El grado de conflicto entre
    dos opiniones es DC = PD * (1-u_a)(1-u_b), con PD la distancia proyectada
    0.5*sum|P_a - P_b| y P = b + u/K. Cada opinión se descuenta por
    eta_m = mean_{n != m}(1 - DC(m, n)) (b' = eta b, u' = 1 - sum b'), y las opiniones
    descontadas se fusionan por promedio de creencias (averaging belief fusion).
    Pérdida: log-verosimilitud de Dirichlet esperada (CE digamma) de cada opinión
    y de la fusión. La probabilidad final es P = b + u/K de la opinión fusionada."""

    def __init__(self, dims: list[int], hidden=64, dropout=0.1, k=NUM_CLASSES):
        super().__init__()
        self.k = k
        self.heads = nn.ModuleList([MLP(d, hidden, dropout, k) for d in dims])

    def opinions(self, xs):
        out = []
        for head, x in zip(self.heads, xs):
            e = F.softplus(head(x))
            s = e.sum(-1, keepdim=True) + self.k
            out.append((e / s, self.k / s, e + 1))
        return out

    def fuse(self, ops):
        m = len(ops)
        probs = [b + u / self.k for b, u, _ in ops]
        disc = []
        for i in range(m):
            agree = []
            for j in range(m):
                if i != j:
                    pd = 0.5 * (probs[i] - probs[j]).abs().sum(-1, keepdim=True)
                    dc = pd * (1 - ops[i][1]) * (1 - ops[j][1])
                    agree.append(1 - dc)
            disc.append(torch.stack(agree, 0).mean(0) if agree else torch.ones_like(ops[i][1]))
        bs = [d * b for d, (b, _, _) in zip(disc, ops)]
        us = [1 - b.sum(-1, keepdim=True) for b in bs]
        # averaging belief fusion for m opinions
        prod_u = torch.stack(us, 0).clamp(min=1e-6)
        denom = sum(torch.prod(torch.cat([prod_u[:j], prod_u[j + 1:]], 0), 0) for j in range(m))
        b = sum(bs[j] * torch.prod(torch.cat([prod_u[:j], prod_u[j + 1:]], 0), 0) for j in range(m)) / denom
        u = m * torch.prod(prod_u, 0) / denom
        return b, u

    def forward(self, xs):
        ops = self.opinions(xs)
        b, u = self.fuse(ops)
        p = (b + u / self.k).clamp(1e-6, 1)
        return torch.log(p / p.sum(-1, keepdim=True)), ops

    @staticmethod
    def dirichlet_loss(alpha, y):
        s = alpha.sum(-1)
        return (torch.digamma(s) - torch.digamma(alpha.gather(1, y.unsqueeze(1)).squeeze(1))).mean()
