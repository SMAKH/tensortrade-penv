import torch
import torch.nn as nn
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


class ReallocationModel(TorchModelV2, nn.Module):
    """A simple model that takes the last n observations as input."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name, **kwargs):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, None, model_config, name)

        m = model_config["custom_model_config"]["num_assets"]
        f = model_config["custom_model_config"]["num_features"]
        second_channels = model_config["custom_model_config"]["second_channels"]
        third_channels = model_config["custom_model_config"]["third_channels"]
        forth_channels = model_config["custom_model_config"]["forth_channels"]
        f = 9396
        self.a = nn.LSTM(f, 16)
        self.b = nn.Linear(16, 128)
        self.c = nn.ReLU()
        self.d = nn.Linear(128, 3240)
        self.e = nn.ReLU()
        self.f = nn.Flatten()
        # self.conv = nn.Sequential(
        #     nn.LSTM(f, 16),
        #     nn.Linear(16, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3240),
        #     nn.ReLU(),
        #     nn.Flatten()
        # )

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

        self.policy_head = nn.Linear(3240 + (m + 1), m + 1)
        self.value_head = nn.Linear(3240 + (m + 1), 1)

        self._last_value = None

        self.view_requirements["prev_actions"] = ViewRequirement(
            shift=-1,
            data_col="actions",
            space=action_space
        )

    def forward(self, input_dict, states, seq_lens):
        # obs = input_dict["obs_flat"]
        weights = input_dict["prev_actions"]
        obs = input_dict["obs"]
        # obs = torch.transpose(obs, 1, 2)
        obs = torch.flatten(obs, 2)

        print("-----------OBS=----------------\n")
        print(obs.shape)
        A = self.a(obs)
        print("--------OBS DONE--------\n")

        # print(A)
        print(A[0].shape)
        B = self.b(A)

        print(B.shape)
        C = self.c(B)

        print(C.shape)
        D = self.d(C)

        print(D.shape)
        E = self.e(D)


        print(E.shape)
        H = self.f(E)

        print(H.shape)
        print('\n------------------\n')
        # H = self.conv(obs)
        X = torch.cat([H, weights], dim=1)
        logits = self.policy_head(X)
        self._last_value = self.value_head(X)

        logits[logits != logits] = 1
        return torch.tanh(logits), []

    def value_function(self):
        return torch.squeeze(self._last_value, -1)
