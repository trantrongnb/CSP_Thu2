import torch
from torch.nn import functional as F

from .temporal_pool import masked_softmax


class RelationNetwork(torch.nn.Module):
    """Learned pair scorer conditioned on Hausdorff and support-set distances."""

    def __init__(self, dimension=128, dropout=0.1):
        super().__init__()
        hidden = max(dimension // 2, 16)
        self.scorer = torch.nn.Sequential(
            torch.nn.Linear(4 * dimension + 3, dimension),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dimension, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(self, query, prototype, set_distance_features):
        pair_features = torch.cat(
            [query, prototype, query - prototype, query * prototype,
             set_distance_features],
            dim=-1,
        )
        return self.scorer(pair_features).squeeze(-1)


def _relation_matrix(queries, prototypes, distance_features, relation_network):
    query_pairs = queries[:, None, :].expand(-1, prototypes.shape[0], -1)
    prototype_pairs = prototypes[None, :, :].expand(queries.shape[0], -1, -1)
    return relation_network(query_pairs, prototype_pairs, distance_features)


def _support_distance_features(query, support, support_valid=None):
    """Return max (Hausdorff), mean, and nearest support distances per pair.

    Queries are singleton sets; support retains the individual examples instead
    of being collapsed into a mean prototype. Leading axes are query, class,
    optional support-shot, then feature dimensions. The three statistics help
    the relation scorer distinguish a broad support set from a single outlier.
    """
    distances = torch.linalg.vector_norm(
        query.unsqueeze(1).unsqueeze(2) - support.unsqueeze(0), dim=-1
    )
    if support_valid is None:
        support_valid = torch.ones(support.shape[:-1], dtype=torch.bool,
                                   device=support.device)
    valid = support_valid.unsqueeze(0)
    nearest = distances.masked_fill(~valid, torch.inf).amin(dim=2)
    farthest = distances.masked_fill(~valid, -torch.inf).amax(dim=2)
    mean = distances.masked_fill(~valid, 0).sum(dim=2) / support_valid.sum(dim=1).clamp_min(1).unsqueeze(0)
    has_support = support_valid.any(dim=1).unsqueeze(0)
    features = torch.stack([farthest, mean, nearest], dim=-1)
    return torch.where(has_support.unsqueeze(-1), features, torch.zeros_like(features))


def part_proto_logits(sg, sp, sq, qg, qp, qq, lam=1.0, beta_d=1.0,
                      beta_q=0.5, tau_a=1.0, rho=0.8, tau_s=0.2, eps=1e-6,
                      adaptive=True, detach_alpha=False, matcher="euclidean",
                      relation_network=None):
    """Support-only prototypes/alpha; query labels never enter the head."""
    if tau_a <= 0 or tau_s <= 0 or not 0 <= rho <= 1 or lam < 0:
        raise ValueError("Require positive temperatures, rho in [0,1], lambda >= 0")
    if matcher not in {"euclidean", "relation", "hausdorff"}:
        raise ValueError("matcher must be 'euclidean', 'relation', or 'hausdorff'")
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
    elif matcher == "relation":
        # Supply Hausdorff, mean, and nearest-shot distances so the learned
        # scorer can use set spread without letting one distant shot dominate.
        global_set_distances = _support_distance_features(qg, sg)
        part_set_distances = _support_distance_features(qp, sp, sq > eps)
        global_score = _relation_matrix(qg, pg, global_set_distances, relation_network)
        query_count = qp.shape[0]
        query_pairs = qp[:, None, :, :].expand(-1, pp.shape[0], -1, -1)
        prototype_pairs = pp[None, :, :, :].expand(query_count, -1, -1, -1)
        part_score = relation_network(query_pairs, prototype_pairs, part_set_distances)
    else:
        # Direct Hausdorff matching remains available as a fixed-metric baseline.
        global_score = -_support_distance_features(qg, sg)[..., 0]
        part_score = -_support_distance_features(qp, sp, sq > eps)[..., 0]
    logits = (global_score + lam * (weight * part_score).sum(-1)) / tau_s
    return logits, alpha
