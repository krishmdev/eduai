"""Online Elo-style calibration on the shared response model.

theta <- theta + K_u (y - p),  b <- b - K_i (y - p),  K = K0 / (1 + a n)
where p is `irt.p_correct`, i.e. the same guessing-floor likelihood the assessment uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eduai.kt.irt import GUESS, p_correct


@dataclass
class EloConfig:
    k0_user: float = 0.8
    a_user: float = 0.05
    k0_item: float = 0.6
    a_item: float = 0.02
    c: float = GUESS


@dataclass
class EloModel:
    cfg: EloConfig = field(default_factory=EloConfig)
    theta: dict[str, float] = field(default_factory=dict)
    b: dict[str, float] = field(default_factory=dict)
    n_user: dict[str, int] = field(default_factory=dict)
    n_item: dict[str, int] = field(default_factory=dict)

    def k_user(self, key: str) -> float:
        return self.cfg.k0_user / (1.0 + self.cfg.a_user * self.n_user.get(key, 0))

    def k_item(self, item: str) -> float:
        return self.cfg.k0_item / (1.0 + self.cfg.a_item * self.n_item.get(item, 0))

    def ensure_item(self, item: str, b0: float) -> None:
        self.b.setdefault(item, b0)

    def predict(self, user_key: str, item: str) -> float:
        return float(p_correct(self.theta.get(user_key, 0.0), self.b.get(item, 0.0), self.cfg.c))

    def update(
        self, user_key: str, item: str, correct: bool, *, update_user: bool = True, update_item: bool = True
    ) -> float:
        p = self.predict(user_key, item)
        err = (1.0 if correct else 0.0) - p
        if update_user:
            self.theta[user_key] = self.theta.get(user_key, 0.0) + self.k_user(user_key) * err
            self.n_user[user_key] = self.n_user.get(user_key, 0) + 1
        if update_item:
            self.b[item] = self.b.get(item, 0.0) - self.k_item(item) * err
            self.n_item[item] = self.n_item.get(item, 0) + 1
        return p
