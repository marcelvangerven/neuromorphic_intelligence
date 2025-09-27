import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import equinox as eqx
from dataclasses import field
from jaxtyping import Array, PyTree
from typing import Callable
from neuromorphic_intelligence.utils import MixedLinearOperator
from abc import ABC, abstractmethod, abstractproperty

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

class StateSpaceModel(eqx.Module):

    @property
    def initial(self):
        return 0.0

    def drift(self, t, x, args):
        return 0.0

    def diffusion(self, t, x, args):
        return None

    @property
    def noise_shape(self):
        return None

    def output(self, t, x, args):
        return x

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea)
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise))


class ParameterizedModel(StateSpaceModel):
    # a parameterized state space model where x = (state, parameters)

    @abstractproperty
    def parameters(self):
        # returns the model parameters
        pass

    @property
    def initial(self):
        return (0.0, self.parameters)

    def output(self, t, x, args):
        # returns the state where x = (state, parameters)
        (state, parameters) = x
        return state

    def value(self, t, x, args):
        # optional value output for critic
        0.0

class OUAModel(eqx.Module):

    model: ParameterizedModel = field(default_factory=None)

    param_rate: float = field(default=1.0) # rate hyperparameter of parameters theta (lambda in OUA paper)
    noise_rate: float = field(default=0.1) # standard deviation of the noise (sigma in OUA paper)
    mean_rate: float = field(default=4.0) # rate parameter of mu (learning rate) (eta in OUA paper)
    reward_rate: float = field(default=2.0) # rate parameter of the reward (rho in OUA paper)

    @property
    def initial(self):
        # state, parameters, means, avg_reward
        return self.model.initial + (self.model.parameters,) + (0.0,)

    def drift(self, t, x, args):
        state, parameters, means, avg_reward = x

        reward = args['reward'](t, x, args)
        RPE = reward - avg_reward

        state_drift = self.model.drift(t, (state, parameters), args)
        param_drift = jax.tree.map(lambda _theta, _mu: self.param_rate * (_mu - _theta), parameters, means)
        mean_drift = jax.tree.map(lambda _mu, _theta: self.mean_rate * RPE * (_theta - _mu), means, parameters)
        return (state_drift, param_drift, mean_drift, self.reward_rate * RPE)

    def diffusion(self, t, x, args):
        state, parameters, means, avg_reward = x

        state_diffusion = self.model.diffusion(t, (state, parameters), args)
        param_diffusion = jax.tree.map(lambda x: self.noise_rate, parameters)
        mean_diffusion = jax.tree.map(lambda x: None, means)

        pytree = (state_diffusion, param_diffusion, mean_diffusion, None)
        return MixedLinearOperator(pytree, input_structure=self.noise_shape)

    def output(self, t, x, args):
        state, parameters, means, avg_reward = x
        return self.model.output(t, (state, parameters), args)

    @property
    def noise_shape(self):
        param_noise_shape = jax.tree.map(lambda x: jax.ShapeDtypeStruct(shape=(x.shape if isinstance(x, Array) else ()), dtype=default_float), self.model.parameters)
        return (self.model.noise_shape, param_noise_shape, param_noise_shape, None)

    def terms(self, key):
        process_noise = dfx.UnsafeBrownianPath(shape=self.noise_shape, key=key, levy_area=dfx.SpaceTimeLevyArea)
        return dfx.MultiTerm(dfx.ODETerm(self.drift), dfx.ControlTerm(self.diffusion, process_noise))

