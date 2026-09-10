from __future__ import annotations

import numpy as np


class DPOTrainer:
    """
    Direct Preference Optimization (DPO) Loss & Gradient Engine.
    Aligns language models directly on human preference pairs (y_w > y_l)
    without needing an explicit reward model.

    Parameters
    ----------
    beta : float, default=0.1
        Temperature parameter scaling implicit reward differences.
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def compute_loss(
        self,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
        ref_chosen_logps: np.ndarray,
        ref_rejected_logps: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Bradley-Terry DPO loss and implicit rewards.

        Formula:
        L_DPO = -log(sigmoid(beta * (log(pi_theta(y_w|x) / pi_ref(y_w|x)) - log(pi_theta(y_l|x) / pi_ref(y_l|x)))))

        Returns
        -------
        loss : float
            Mean batch DPO loss.
        chosen_rewards : np.ndarray
            Implicit rewards for chosen responses.
        rejected_rewards : np.ndarray
            Implicit rewards for rejected responses.
        """
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = self.beta * (pi_logratios - ref_logratios)

        # Stable log(sigmoid(logits))
        # log(1 / (1 + exp(-logits))) = -softplus(-logits)
        loss = float(np.mean(np.log1p(np.exp(-np.clip(logits, -30.0, 30.0)))))

        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)

        return loss, chosen_rewards, rejected_rewards


class KTOTrainer:
    """
    Kahneman-Tversky Optimization (KTO) Engine for Unpaired Binary Preferences (Up/Down).

    Parameters
    ----------
    beta : float, default=0.1
        Temperature parameter.
    desirable_weight : float, default=1.0
        Weight for desirable examples lambda_D.
    undesirable_weight : float, default=1.0
        Weight for undesirable examples lambda_U.
    """

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ) -> None:
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def compute_loss(
        self,
        policy_logps: np.ndarray,
        ref_logps: np.ndarray,
        labels: np.ndarray,
        kl_ref: float = 0.0,
    ) -> float:
        """
        Compute KTO loss.
        labels: boolean array (True = desirable/chosen, False = undesirable/rejected).
        """
        log_ratios = policy_logps - ref_logps
        implicit_rewards = self.beta * log_ratios - self.beta * kl_ref

        losses = np.zeros_like(policy_logps, dtype=np.float32)
        for i, is_pos in enumerate(labels):
            z = implicit_rewards[i]
            if is_pos:
                # 1 - sigmoid(z)
                losses[i] = self.desirable_weight * float(
                    1.0 / (1.0 + np.exp(np.clip(z, -30.0, 30.0)))
                )
            else:
                # 1 - sigmoid(-z)
                losses[i] = self.undesirable_weight * float(
                    1.0 / (1.0 + np.exp(np.clip(-z, -30.0, 30.0)))
                )

        return float(np.mean(losses))


class ORPOTrainer:
    """
    Odds Ratio Preference Optimization (ORPO) Engine.
    Combines Supervised Fine-Tuning (SFT) with Odds Ratio Penalty without reference model.

    Parameters
    ----------
    lambda_orpo : float, default=0.1
        Weight of Odds Ratio penalty relative to cross-entropy loss.
    """

    def __init__(self, lambda_orpo: float = 0.1) -> None:
        self.lambda_orpo = lambda_orpo

    def compute_loss(
        self,
        sft_loss: float,
        policy_chosen_logps: np.ndarray,
        policy_rejected_logps: np.ndarray,
    ) -> float:
        """Compute ORPO loss: L_SFT + lambda * (-log(sigmoid(odds_ratio)))."""
        # Odds(y|x) = P(y|x) / (1 - P(y|x)) -> log Odds = log P - log(1 - P)
        p_chosen = np.exp(np.clip(policy_chosen_logps, -30.0, 0.0))
        p_rejected = np.exp(np.clip(policy_rejected_logps, -30.0, 0.0))

        log_odds_chosen = policy_chosen_logps - np.log1p(
            -np.clip(p_chosen, 0.0, 0.9999)
        )
        log_odds_rejected = policy_rejected_logps - np.log1p(
            -np.clip(p_rejected, 0.0, 0.9999)
        )

        log_odds_ratio = log_odds_chosen - log_odds_rejected
        odds_loss = np.mean(np.log1p(np.exp(-np.clip(log_odds_ratio, -30.0, 30.0))))

        return float(sft_loss + self.lambda_orpo * odds_loss)
