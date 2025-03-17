import torch
import torch.nn as nn
from torch.distributions import Categorical
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomCnnPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(CustomCnnPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            ortho_init=True,
            **kwargs
        )
        # Define the CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Input: (batch, 1, 6, 7) -> (batch, 32, 6, 7)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: (batch, 64, 6, 7)
            nn.ReLU(),
            nn.Flatten(),  # Output: batch * (64 * 6 * 7) = batch * 2688
            nn.Linear(2688, 128),
            nn.ReLU()
        )
        self.policy_net = nn.Linear(128, action_space.n)
        self.value_net = nn.Linear(128, 1)

    def forward(self, obs, deterministic=False):
        obs = obs.to(torch.float32).permute(0, 3, 1, 2)
        features = self.cnn(obs)
        logits = self.policy_net(features)
        values = self.value_net(features)
        dist = Categorical(logits=logits)
        
        if deterministic:
            actions = torch.argmax(logits, dim=1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        return actions, values, log_probs

    def extract_features(self, obs):
        obs = obs.to(torch.float32).permute(0, 3, 1, 2)
        return self.cnn(obs)

    def evaluate_actions(self, obs, actions):
        obs = obs.to(torch.float32).permute(0, 3, 1, 2)
        features = self.cnn(obs)
        logits = self.policy_net(features)
        values = self.value_net(features)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_probs, entropy

    def predict_values(self, obs):
        obs = obs.to(torch.float32).permute(0, 3, 1, 2)
        features = self.cnn(obs)
        return self.value_net(features)