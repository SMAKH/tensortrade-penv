import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.view_requirement import ViewRequirement


class Dirichlet(TorchDistributionWrapper):

    def __init__(self, inputs, model):
        self.epsilon = torch.tensor(1e-7).to(inputs.device)
        concentration = torch.exp(inputs) + self.epsilon
        self.dist = torch.distributions.dirichlet.Dirichlet(
            concentration=concentration,
            validate_args=True
        )
        super().__init__(concentration, model)

    def deterministic_sample(self):
        self.last_sample = torch.softmax(self.dist.concentration, dim=-1)
        return self.last_sample

    def logp(self, x):
        x = torch.max(x, self.epsilon)
        x = x / torch.sum(x, dim=-1, keepdim=True)
        return self.dist.log_prob(x)

    def entropy(self):
        return self.dist.entropy()

    def kl(self, other):
        return torch.distributions.kl.kl_divergence(self.dist, other.dist)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return action_space.m + 1


class ReallocationModel(RecurrentNetwork, nn.Module):
    """A simple model that takes the last n observations as input."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, None, model_config, name)

        self.cell_size = 128
        m = model_config["custom_model_config"]["num_assets"]
        # f = model_config["custom_model_config"]["num_features"]
        second_channels = model_config["custom_model_config"]["second_channels"]
        third_channels = model_config["custom_model_config"]["third_channels"]
        forth_channels = model_config["custom_model_config"]["forth_channels"]
        f = 9396

        self.a = nn.LSTM(f, self.cell_size)
        self.b = nn.Linear(self.cell_size, 256)
        self.c = nn.ReLU()
        self.d = nn.Linear(256, 512)
        self.e = nn.ReLU()
        self.f = nn.Flatten()


        ### out = 3240
        # self.conv = nn.Sequential(nn.Conv2d(f, second_channels, (3, 1), stride=(2, 1), bias=True),
        #                           nn.ReLU(),
        #                           nn.Conv2d(second_channels, third_channels, (3, 1), stride=(2, 1), bias=True),
        #                           # nn.BatchNorm2d(third_channels),
        #                           nn.ReLU(),
        #                           nn.Conv2d(third_channels, forth_channels, (3, 1), stride=(2, 1), bias=True),
        #                           nn.ReLU(),
        #                           nn.Flatten()
        #                           )

        self.policy_head = nn.Linear(512 + (m + 1), m + 1)
        self.value_head = nn.Linear(512 + (m + 1), 1)

        self._last_value = None

        self.view_requirements["prev_actions"] = ViewRequirement(
            shift=-1,
            data_col="actions",
            space=action_space
        )

    def forward(self, input_dict, states, seq_lens):
        print(states)
        print("------------END------------")
        # obs = input_dict["obs_flat"]
        weights = input_dict["prev_actions"]
        obs = input_dict["obs"]
        # obs = torch.transpose(obs, 1, 2)
        obs = torch.flatten(obs, 2)
        obs = obs[:,-1,:]
        states = torch.transpose(states, 0, 1)


        A, newstate = self.a(obs, states)
        print("------------START------------")
        print(newstate)

        B = self.b(A)
        C = self.c(B)
        D = self.d(C)
        E = self.e(D)
        H = self.f(E)

        # H = self.conv(obs)
        X = torch.cat([H, weights], dim=1)
        logits = self.policy_head(X)
        self._last_value = self.value_head(X)

        logits[logits != logits] = 1
        return torch.tanh(logits), [newstate[0], newstate[1]]

    def get_initial_state(self):
        return [
            torch.tensor(np.zeros(self.cell_size, np.float32)),
            torch.tensor(np.zeros(self.cell_size, np.float32))]
    def value_function(self):
        return torch.squeeze(self._last_value, -1)
