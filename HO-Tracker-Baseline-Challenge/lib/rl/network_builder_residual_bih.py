import math
import torch
import torch.nn as nn
from hydra.utils import instantiate
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder
from .network_builder import DictObsNetwork, DictObsBuilder
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.model_builder import NetworkBuilder as ModelNetworkBuilder
from lib.utils.torch_utils import recurse_freeze, freeze_batchnorm_stats
from lib.rl.models import BaseModelNetwork


class ResBiHDictObsNetwork(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop("actions_num")
        self.value_size = kwargs.pop("value_size", 1)
        self.num_seqs = kwargs.pop("num_seqs", 1)

        NetworkBuilder.BaseNetwork.__init__(self)
        self.load(params)
        self.actor_cnn = nn.Sequential()
        self.critic_cnn = nn.Sequential()
        self.actor_mlp = nn.Sequential()
        self.critic_mlp = nn.Sequential()

        self.dict_feature_encoder = instantiate(params["dict_feature_encoder"])
        mlp_input_shape = self.dict_feature_encoder.output_dim

        in_mlp_shape = mlp_input_shape
        if len(self.units) == 0:
            out_size = mlp_input_shape
        else:
            out_size = self.units[-1]

        mlp_args = {
            "input_size": in_mlp_shape,
            "units": self.units,
            "activation": self.activation,
            "norm_func_name": self.normalization,
            "dense_func": torch.nn.Linear,
            "d2rl": self.is_d2rl,
            "norm_only_first_layer": self.norm_only_first_layer,
        }
        self.actor_mlp = self._build_mlp(**mlp_args)
        if self.separate:
            raise NotImplementedError("separate not implemented")

        self.value = self._build_value_layer(out_size, self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)

        if self.is_discrete:
            self.logits = torch.nn.Linear(out_size, actions_num)
        """
            for multidiscrete actions num is a tuple
        """
        if self.is_multi_discrete:
            self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
        if self.is_continuous:
            self.mu = torch.nn.Linear(out_size, actions_num)
            self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
            mu_init = self.init_factory.create(**self.space_config["mu_init"])
            self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
            sigma_init = self.init_factory.create(**self.space_config["sigma_init"])

            if self.fixed_sigma:
                self.sigma = nn.Parameter(
                    torch.zeros(actions_num, requires_grad=True, dtype=torch.float32),
                    requires_grad=True,
                )
            else:
                self.sigma = torch.nn.Linear(out_size, actions_num)

        mlp_init = self.init_factory.create(**self.initializer)
        if self.has_cnn:
            cnn_init = self.init_factory.create(**self.cnn["initializer"])

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                cnn_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                mlp_init(m.weight)
                if getattr(m, "bias", None) is not None:
                    torch.nn.init.zeros_(m.bias)

        if self.is_continuous:
            mu_init(self.mu.weight)
            # nn.init.xavier_normal_(self.mu.weight, 0.0001)
            # nn.init.zeros_(self.mu.bias)
            if self.fixed_sigma:
                sigma_init(self.sigma)
            else:
                sigma_init(self.sigma.weight)

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        states = obs_dict.get("rnn_states", None)

        obs = self.dict_feature_encoder(obs)

        if self.separate:
            raise NotImplementedError("separate not implemented")
        else:
            out = obs
            out = self.actor_cnn(out)
            out = out.flatten(1)

            out = self.actor_mlp(out)
            value = self.value_act(self.value(out))

            if self.central_value:
                raise NotImplementedError("central_value not implemented")

            if self.is_discrete:
                raise NotImplementedError("is_discrete not implemented")
            if self.is_multi_discrete:
                raise NotImplementedError("is_multi_discrete not implemented")
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return (
                    torch.nan_to_num(mu),
                    torch.nan_to_num(mu * 0 + sigma),
                    torch.nan_to_num(value),
                    states,
                )


class ResBiHDictObsBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        net = ResBiHDictObsNetwork(self.params, **kwargs)
        return net
