"""Knowledge Graph Embeddings (RotatE and TransE) in pure NumPy."""

from __future__ import annotations


import numpy as np


class RotatE:
    r"""RotatE Knowledge Graph Embedding Model in Complex Space (Sun et al. 2019).

    Models relational triplets $(h, r, t)$ as elementwise complex rotations:
    $\mathbf{t} \approx \mathbf{h} \circ \mathbf{r}$, capturing symmetric,
    anti-symmetric, inverse, and composition relational patterns.

    Parameters
    ----------
    num_entities : int
        Total number of entities in knowledge graph.
    num_relations : int
        Total number of relations in knowledge graph.
    embedding_dim : int, default=64
        Complex embedding dimensionality (number of complex numbers).
    gamma : float, default=12.0
        Margin parameter for distance scoring.
    seed : int, default=42
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        gamma: float = 12.0,
        seed: int = 42,
    ) -> None:
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.gamma = float(gamma)

        rng = np.random.default_rng(seed)
        scale = gamma / embedding_dim

        # Entity complex embeddings (Real + Imaginary parts): (N, D)
        self.ent_re: np.ndarray = rng.uniform(
            -scale, scale, size=(num_entities, embedding_dim)
        ).astype(np.float32)
        self.ent_im: np.ndarray = rng.uniform(
            -scale, scale, size=(num_entities, embedding_dim)
        ).astype(np.float32)

        # Relation phase angles theta in [-pi, pi]: (R, D)
        self.rel_phase: np.ndarray = rng.uniform(
            -np.pi, np.pi, size=(num_relations, embedding_dim)
        ).astype(np.float32)

    def score_triplets(
        self, head_ids: np.ndarray, rel_ids: np.ndarray, tail_ids: np.ndarray
    ) -> np.ndarray:
        """Computes distance score for given knowledge graph triplets (h, r, t).

        Score: gamma - || h * r - t ||_2
        """
        h_ids = np.asarray(head_ids, dtype=np.int64)
        r_ids = np.asarray(rel_ids, dtype=np.int64)
        t_ids = np.asarray(tail_ids, dtype=np.int64)

        h_re, h_im = self.ent_re[h_ids], self.ent_im[h_ids]
        t_re, t_im = self.ent_re[t_ids], self.ent_im[t_ids]

        # Unit complex rotation: r = cos(theta) + i sin(theta)
        r_re = np.cos(self.rel_phase[r_ids])
        r_im = np.sin(self.rel_phase[r_ids])

        # Complex multiplication: h * r = (h_re * r_re - h_im * r_im) + i (h_re * r_im + h_im * r_re)
        prod_re = h_re * r_re - h_im * r_im
        prod_im = h_re * r_im + h_im * r_re

        # Distance: || h * r - t ||
        diff_re = prod_re - t_re
        diff_im = prod_im - t_im

        dist = np.sqrt(np.sum(diff_re**2 + diff_im**2, axis=-1) + 1e-12)
        return self.gamma - dist


class TransE:
    r"""TransE Knowledge Graph Translation Embedding Model (Bordes et al. 2013).

    Models relations as translations in vector space: $\mathbf{h} + \mathbf{r} \approx \mathbf{t}$.

    Parameters
    ----------
    num_entities : int
        Total number of entities.
    num_relations : int
        Total number of relations.
    embedding_dim : int, default=64
        Vector embedding dimensionality.
    margin : float, default=1.0
        Margin parameter for ranking loss.
    seed : int, default=42
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 64,
        margin: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.embedding_dim = int(embedding_dim)
        self.margin = float(margin)

        rng = np.random.default_rng(seed)
        scale = 6.0 / np.sqrt(embedding_dim)

        self.entity_embeddings: np.ndarray = rng.uniform(
            -scale, scale, size=(num_entities, embedding_dim)
        ).astype(np.float32)
        self.relation_embeddings: np.ndarray = rng.uniform(
            -scale, scale, size=(num_relations, embedding_dim)
        ).astype(np.float32)

        # Normalize relations
        norms = np.linalg.norm(self.relation_embeddings, axis=-1, keepdims=True) + 1e-12
        self.relation_embeddings /= norms

    def score_triplets(
        self, head_ids: np.ndarray, rel_ids: np.ndarray, tail_ids: np.ndarray
    ) -> np.ndarray:
        """Computes translation score: -|| h + r - t ||_2."""
        h_ids = np.asarray(head_ids, dtype=np.int64)
        r_ids = np.asarray(rel_ids, dtype=np.int64)
        t_ids = np.asarray(tail_ids, dtype=np.int64)

        h = self.entity_embeddings[h_ids]
        r = self.relation_embeddings[r_ids]
        t = self.entity_embeddings[t_ids]

        dist = np.linalg.norm(h + r - t, axis=-1)
        return -dist

    def margin_loss(
        self,
        pos_heads: np.ndarray,
        pos_rels: np.ndarray,
        pos_tails: np.ndarray,
        neg_heads: np.ndarray,
        neg_rels: np.ndarray,
        neg_tails: np.ndarray,
    ) -> float:
        """Calculates margin-based ranking loss: max(0, margin - pos_score + neg_score)."""
        pos_scores = self.score_triplets(pos_heads, pos_rels, pos_tails)
        neg_scores = self.score_triplets(neg_heads, neg_rels, neg_tails)

        loss = np.maximum(0.0, self.margin - pos_scores + neg_scores)
        return float(np.mean(loss))
