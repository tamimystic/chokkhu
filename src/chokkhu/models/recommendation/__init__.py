"""Chokkhu Recommendation & Collaborative Filtering Subsystem.

Sovereign pure NumPy implementations of classic and modern deep recommendation systems:
- Matrix Factorization: SVD, SVD++, ImplicitALS, NMFRecommender
- Pairwise Ranking: BayesianPersonalizedRanking (BPR-MF)
- Deep Recommenders: NeuralCollaborativeFiltering (NCF), WideAndDeep, DeepFM, DLRM
- Sequential Recommenders: SASRec, GRU4Rec
- Evaluation Metrics: hit_rate_at_k, ndcg_at_k, mrr_at_k, precision_at_k, recall_at_k, map_at_k
"""

from chokkhu.models.recommendation.matrix_factorization import (
    ImplicitALS,
    NMFRecommender,
    SVDPlusPlus,
    SVDRecommender,
)
from chokkhu.models.recommendation.bpr import BayesianPersonalizedRanking
from chokkhu.models.recommendation.ncf import NeuralCollaborativeFiltering
from chokkhu.models.recommendation.wide_and_deep import WideAndDeep
from chokkhu.models.recommendation.deepfm import DeepFM
from chokkhu.models.recommendation.dlrm import DLRM
from chokkhu.models.recommendation.sequential import SASRec, GRU4Rec
from chokkhu.models.recommendation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    map_at_k,
)

__all__ = [
    "SVDRecommender",
    "SVDPlusPlus",
    "ImplicitALS",
    "NMFRecommender",
    "BayesianPersonalizedRanking",
    "NeuralCollaborativeFiltering",
    "WideAndDeep",
    "DeepFM",
    "DLRM",
    "SASRec",
    "GRU4Rec",
    "hit_rate_at_k",
    "ndcg_at_k",
    "mrr_at_k",
    "precision_at_k",
    "recall_at_k",
    "map_at_k",
]
