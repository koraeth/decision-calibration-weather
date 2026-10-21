"""Decision tasks.

A task is any object with
    n_actions:  number of actions, indexed 0 .. n_actions - 1
    cost(a, y): cost of action index a for outcome y; a (int or int array) and y broadcast
"""
from dataclasses import dataclass

import numpy as np


def _check_actions(a, n_actions):
    """a as integer array; raises for anything but indices 0 .. n_actions - 1 (np.where / indexing would not)."""
    a = np.asarray(a)
    if not (np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a < n_actions))):
        raise ValueError(f'actions must be integer indices 0 .. {n_actions - 1}')
    return a


@dataclass
class FrostTask:
    """Frost protection, y = 2 m temperature (degC). a = 0: no protection, a = 1: protect.

    Cost ramps linearly over [threshold, threshold + gap_size]: missed frost costs up to
    10 * cost_ratio, needless protection up to 10 * (1 - cost_ratio).
    """
    threshold: float
    cost_ratio: float
    gap_size: float = 3.0
    n_actions = 2

    def cost(self, a, y):
        a = _check_actions(a, self.n_actions)
        ramp = np.clip((np.asarray(y, dtype=float) - self.threshold) / self.gap_size, 0, 1)
        return np.where(a == 1,
                        (1 - self.cost_ratio) * 10 * ramp,
                        self.cost_ratio * 10 * (1 - ramp))


@dataclass
class HeatTask:
    """Heat protection, y = 2 m temperature (degC). a = 0: no protection, a = 1: protect.

    Cost ramps linearly over [threshold - gap_size, threshold]: missed heat costs up to
    10 * cost_ratio, needless protection up to 10 * (1 - cost_ratio).
    """
    threshold: float
    cost_ratio: float
    gap_size: float = 3.0
    n_actions = 2

    def cost(self, a, y):
        a = _check_actions(a, self.n_actions)
        y = np.asarray(y, dtype=float)
        # two ramps written separately (not 1 - ramp) to keep the paper's floating-point results
        return np.where(a == 1,
                        (1 - self.cost_ratio) * 10 * np.clip((self.threshold - y) / self.gap_size, 0, 1),
                        np.clip((y - (self.threshold - self.gap_size)) / self.gap_size, 0, 1) * self.cost_ratio * 10)


@dataclass
class WindPowerTask:
    """Day-ahead bid of wind power, y = normalised power output in [0, 1]. a = index into bids.

    Cost = -profit. Profit = bid, minus shortfall_penalty per unit below the bid and
    surplus_penalty per unit above it. Bid 0 (turbine off) costs zero_bid_cost (standby drain).
    """
    shortfall_penalty: float
    surplus_penalty: float = 0.0
    zero_bid_cost: float = 0.01
    bids: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    @property
    def n_actions(self):
        return len(self.bids)

    def cost(self, a, y):
        a = _check_actions(a, self.n_actions)
        y = np.asarray(y, dtype=float)
        bid = np.asarray(self.bids)[a]
        profit = np.where(y > bid,
                          bid - self.surplus_penalty * (y - bid),
                          bid - self.shortfall_penalty * (bid - y))
        return np.where(bid == 0, self.zero_bid_cost, -profit)
