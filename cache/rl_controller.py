"""
TurboCache RL Controller: Hybrid RL Agent (Stage 3)
Learns optimal cache eviction nudges using policy gradients / Q-learning
with the deterministic heuristic as a safety net.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import random
from typing import Dict, Tuple, List, Optional

class CacheRLPolicy(nn.Module):
    def __init__(self, state_dim: int = 6, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Outputs nudge in range [-1.0, 1.0]
        )

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.net(state_tensor)

class CacheRLAgent:
    """
    Stage 3 RL Agent:
    - Receives state: [layer_id/40, cache_occupancy, recency, frequency, load_ms/10, vram_mb/1000]
    - Computes rl_nudge in [-1.0, 1.0]
    - Receives rewards: +1.0 (GPU hit), +0.5 (RAM hit), -1.0 (SSD load), -2.0 (Bad Eviction)
    - Updates policy parameters online via policy gradient
    """
    def __init__(self, lr: float = 0.005, gamma: float = 0.95):
        self.device = torch.device("cpu")  # CPU execution to avoid GPU lock contention
        self.policy = CacheRLPolicy().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        self.saved_log_probs = []
        self.rewards = []
        self.total_reward = 0.0
        self.updates_count = 0

    def select_nudge_batch(
        self,
        candidate_states: List[Tuple[int, float, float, float, float, float]]
    ) -> List[float]:
        """
        Evaluates a batch of candidate expert states in 1 single vectorized PyTorch forward pass.
        candidate_states: list of (layer_id, cache_occupancy, recency, frequency, load_ms, vram_mb)
        """
        if not candidate_states:
            return []
            
        states_tensor = torch.tensor(
            [[s[0]/40.0, s[1], s[2], s[3], s[4]/10.0, s[5]/1000.0] for s in candidate_states],
            dtype=torch.float32,
            device=self.device
        )
        
        with torch.no_grad():
            nudges = self.policy(states_tensor).squeeze(-1)
            
        return nudges.tolist()

    def reward(self, reward_val: float):
        self.rewards.append(reward_val)
        self.total_reward += reward_val
        
        # Periodically update policy online
        if len(self.rewards) >= 20:
            self.update_policy()

    def update_policy(self):
        if not self.saved_log_probs or not self.rewards:
            return
            
        returns = []
        r_accum = 0
        for r in reversed(self.rewards):
            r_accum = r + self.gamma * r_accum
            returns.insert(0, r_accum)
            
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if returns_t.std() > 1e-5:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-5)
            
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns_t):
            policy_loss.append(-log_prob * R)
            
        self.optimizer.zero_grad()
        if policy_loss:
            loss = torch.cat(policy_loss).sum()
            loss.backward()
            self.optimizer.step()
            self.updates_count += 1
            
        self.saved_log_probs.clear()
        self.rewards.clear()

    def get_summary(self) -> dict:
        return {
            "rl_updates": self.updates_count,
            "cumulative_reward": f"{self.total_reward:.2f}"
        }
