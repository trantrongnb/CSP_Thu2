import torch
from torch.nn import functional as F

from .temporal_pool import masked_softmax


class RelationNetwork(torch.nn.Module):
    """Learned pairwise scorer for query/prototype feature vectors."""

    def __init__(self, dimension=128, dropout=0.1):
        super().__init__()
        hidden = max(dimension // 2, 16)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(4 * dimension, dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dimension, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, query, prototype):
        pair_features = torch.cat(
            [query, prototype, query - prototype, query * prototype], dim=-1
        )
        return self.scorer(pair_features).squeeze(-1)


def _relation_matrix(queries, prototypes, relation_network):
    query_pairs = queries[:, None, :].expand(-1, prototypes.shape[0], -1)
    prototype_pairs = prototypes[None, :, :].expand(queries.shape[0], -1, -1)
    return relation_network(query_pairs, prototype_pairs)


def part_proto_logits(sg, sp, sq, qg, qp, qq, lam=1.0, beta_d=1.0,
                      beta_q=0.5, tau_a=1.0, rho=0.8, tau_s=0.2, eps=1e-6,
                      adaptive=True, detach_alpha=False, matcher="euclidean",
                      relation_network=None):
    """Support-only prototypes/alpha; query labels never enter the head."""
    if tau_a <= 0 or tau_s <= 0 or not 0 <= rho <= 1 or lam < 0:
        raise ValueError("Require positive temperatures, rho in [0,1], lambda >= 0")
    if matcher not in {"euclidean", "relation"}:
        raise ValueError("matcher must be 'euclidean' or 'relation'")
    if matcher == "relation" and relation_network is None:
        raise ValueError("relation_network is required when matcher='relation'")
    n, k, r, d = sp.shape
    if n < 2 or k < 1 or r != 3 or sg.shape != (n, k, d) or sq.shape != (n, k, r):
        raise ValueError("Expected N>=2, K>=1, support [N,K,D]/[N,K,3,D]/[N,K,3]")
    if qg.shape != (qp.shape[0], d) or qp.shape[1:] != (r, d) or qq.shape != qp.shape[:2]:
        raise ValueError("Invalid query shapes")
    # Float32 distances/softmax avoid precision loss and epsilon underflow under AMP.
    sg, sp, qg, qp = [F.normalize(x.float(), dim=-1, eps=eps) for x in (sg, sp, qg, qp)]
    sq, qq = sq.float().clamp(0, 1), qq.float().clamp(0, 1)
    pg = F.normalize(sg.mean(1), dim=-1, eps=eps)
    denom = sq.sum(1)
    pp = F.normalize((sp * sq.unsqueeze(-1)).sum(1) / denom.unsqueeze(-1).clamp_min(eps),
                     dim=-1, eps=eps)
    ok = denom > eps
    pairs = ok[:, None] & ok[None, :] & (~torch.eye(n, dtype=torch.bool, device=sp.device))[:, :, None]
    count = pairs.sum(1)
    dissimilarity = (1 - torch.einsum("crd,krd->ckr", pp, pp)).clamp_min(0)
    discrim = (dissimilarity * pairs).sum(1) / count.clamp_min(1)
    eligible = ok & (count > 0) if adaptive else ok
    scores = (beta_d * discrim + beta_q * sq.mean(1).clamp_min(eps).log()) / tau_a
    uniform = eligible.float() / eligible.sum(-1, keepdim=True).clamp_min(1)
    alpha = ((1 - rho) * uniform + rho * masked_softmax(scores, eligible)) if adaptive else uniform
    if detach_alpha:
        alpha = alpha.detach()
    valid = eligible[None, :] & (qq > eps)[:, None, :]
    weight = alpha[None, :] * valid
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(eps)
    if matcher == "euclidean":
        global_score = -((qg[:, None] - pg[None, :]) ** 2).sum(-1)
        part_score = -((qp[:, None] - pp[None, :]) ** 2).sum(-1)
    else:
        global_score = _relation_matrix(qg, pg, relation_network)
        query_count = qp.shape[0]
        query_pairs = qp[:, None, :, :].expand(-1, pp.shape[0], -1, -1)
        prototype_pairs = pp[None, :, :, :].expand(query_count, -1, -1, -1)
        part_score = relation_network(query_pairs, prototype_pairs)
    logits = (global_score + lam * (weight * part_score).sum(-1)) / tau_s
    return logits, alpha
