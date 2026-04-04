from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score

# This is the data class for all the metics 
@dataclass
class AttackEvaluation:
    clean_accuracy: float
    adversarial_accuracy: float
    accuracy_drop: float
    attack_success_rate: float
    avg_l2_perturbation: float
    avg_linf_perturbation: float

# This is the stage B attack simulator 
class StageBAttackSimulator:
    """Gaussian-noise adversarial simulation for tabular classifiers."""

    def __init__(
        self,
        noise_std: float = 0.1, # How spread out the noise should be 
        clip_min: float | None = None, # min noise
        clip_max: float | None = None, # max noise
        random_state: int = 42,
    ) -> None:
        if noise_std <= 0:
            raise ValueError("noise_std must be > 0")

        self.noise_std = noise_std
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)
    # This generates the adversial attacks using random so to the test it adds a random value 
    def generate_adversarial(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_adv, perturbation) where x_adv = x_test + gaussian_noise."""
        x = np.asarray(x_test, dtype=float)
        perturbation = self._rng.normal(loc=0.0, scale=self.noise_std, size=x.shape)
        x_adv = x + perturbation

        if self.clip_min is not None or self.clip_max is not None:
            x_adv = np.clip(x_adv, self.clip_min, self.clip_max)

        return x_adv, perturbation
    # This is used to calculate all the features 
    def evaluate(self, model: Any, x_test: np.ndarray, y_test: np.ndarray) -> AttackEvaluation:
        """Compute robustness metrics for clean vs adversarial samples."""
        x = np.asarray(x_test, dtype=float)
        y = np.asarray(y_test)

        y_clean_pred = model.predict(x)
        x_adv, perturbation = self.generate_adversarial(x)
        y_adv_pred = model.predict(x_adv)

        clean_accuracy = accuracy_score(y, y_clean_pred)
        adversarial_accuracy = accuracy_score(y, y_adv_pred)
        clean_correct = y_clean_pred == y
        flipped_due_to_attack = (y_adv_pred != y) & clean_correct
        
        return AttackEvaluation(
            clean_accuracy=float(clean_accuracy),
            adversarial_accuracy=float(adversarial_accuracy),
            accuracy_drop=float(clean_accuracy - adversarial_accuracy),
            attack_success_rate=float(np.mean(flipped_due_to_attack)),
            avg_l2_perturbation=float(np.mean(np.linalg.norm(perturbation, ord=2, axis=1))),
            avg_linf_perturbation=float(np.mean(np.linalg.norm(perturbation, ord=np.inf, axis=1))),
        )
