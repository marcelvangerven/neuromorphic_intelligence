import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import equinox as eqx
from neuromorphic_intelligence.utils import MixedLinearOperator
from dataclasses import field
from jaxtyping import Array
import lineax as lx
from neuromorphic_intelligence.oua import StateSpaceModel, ParameterizedModel

default_float = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

class StochasticDoubleIntegrator(StateSpaceModel):

    mass: float = 1.0
    damping_factor: float = 1.0
    noise_scale: float = 1e-3

    @property
    def initial(self):
        return jnp.zeros(2)

    def drift(self, t, x, args):
        position, velocity = x

        # return jnp.array([jnp.where(jnp.abs(position) < 1, velocity, 0.0), - (self.damping_factor/self.mass) * velocity])
        return jnp.array([velocity, - (self.damping_factor/self.mass) * velocity])

    def diffusion(self, t, x, args):
        return jnp.array([0.0, self.noise_scale])

    def output(self, t, x, args):
        return x

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(), dtype=default_float)
    
class HarmonicOscillator(StateSpaceModel):

    mass: float = 1.0
    damping_factor: float = 1.0
    noise_scale: float = 1e-3
    spring_constant: float = 1.0

    @property
    def initial(self):
        return jnp.zeros(2)

    def drift(self, t, x, args):
        position, velocity = x

        omega02 = self.spring_constant / self.mass

        return jnp.array([velocity, - (self.damping_factor/self.mass) * velocity - omega02 * position])

    def diffusion(self, t, x, args):
        return jnp.array([0.0, self.noise_scale])

    def output(self, t, x, args):
        return x

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct(shape=(), dtype=default_float)
 
    # mass: float = 1.0
    # damping_factor: float = 1.0
    # noise_scale: float = 0.0
    # spring_constant: float = 1.0

    # @property
    # def initial(self):
    #     return jnp.zeros(2)

    # def drift(self, t, x, args):
    #     position, velocity = x

    #     omega02 = self.spring_constant / self.mass

    #     return jnp.array([velocity, - (self.damping_factor/self.mass) * velocity] - omega02 * position)

    # def diffusion(self, t, x, args):
    #     return jnp.array([0.0, self.noise_scale])

    # def output(self, t, x, args):
    #     return x

    # @property
    # def noise_shape(self):
    #     return jax.ShapeDtypeStruct(shape=(), dtype=default_float)

class CTRNN(ParameterizedModel):

    num_inputs: int = eqx.field(default=None)
    num_neurons: int = eqx.field(default=None)
    num_outputs: int = eqx.field(default=None)

    # parameters to estimate
    tau: Array = field(default=None) # time constant vector
    A: Array = field(default=None) # recurrent matrix
    bias: Array = field(default=None) # bias vector
    B: Array = field(default=None) # input matrix
    C: Array = field(default=None) # output matrix

    # allow for stochastic neurons
    noise_scale: float = 0.0

    def __post_init__(self):

        self.tau = jnp.zeros((self.num_neurons,), default_float)   

        self.A = jnp.zeros((self.num_neurons, self.num_neurons), default_float)
        self.bias = jnp.zeros((self.num_neurons,), default_float)   

        assert self.num_inputs > 0, "The number of inputs must be greater than 0"
        self.B = jnp.zeros((self.num_neurons, self.num_inputs), default_float)

        assert self.num_outputs > 0, "The number of outputs must be greater than 0"
        self.C = jnp.zeros((self.num_outputs, self.num_neurons), default_float)

    @property
    def initial(self):
        return (jnp.zeros((self.num_neurons,), default_float), self.parameters)        

    @property
    def parameters(self): # actor_parameters vs critic_parameters?
        return (self.tau, self.A, self.bias, self.B, self.C)

    def drift(self, t, x, args):
        state, (tau, A, bias, B, C) = x 

        # log transform on time constants to ensure positivity
        decay = 1.0
        return jnp.exp(tau) * (-decay*state + A @ jax.nn.sigmoid(state + bias))

    def diffusion(self, t, x, args):
        return lx.DiagonalLinearOperator(self.noise_scale * jnp.ones(self.num_neurons))
 
    def output(self, t, x, args):
        state, (tau, A, bias, B, C) = x
        return jax.nn.tanh(jnp.squeeze(C @ state)) # NOTE: we limit the control output

    @property
    def noise_shape(self):
        return jax.ShapeDtypeStruct((self.num_neurons,), default_float)


class CoupledSystem(ParameterizedModel):

    agent: ParameterizedModel
    env: StateSpaceModel

    @property
    def initial(self):
        agent_state, agent_params = self.agent.initial
        env_state = self.env.initial
        return (agent_state, env_state), agent_params

    @property
    def parameters(self):
        return self.agent.parameters

    def drift(self, t, x, args):
        (agent_state, env_state), agent_params = x        
        tau, A, bias, B, C = agent_params

        agent_drift = self.agent.drift(t, (agent_state, agent_params), args) + jnp.tanh(B @ self.env.output(t, env_state, args))
        env_drift = self.env.drift(t, env_state, args) + self.agent.output(t, (agent_state, agent_params), args)

        # boundary conditions for the environment
        # pos, vel = env_state
        # drift = self.env.drift(t, env_state, args) + self.agent.output(t, (agent_state, agent_params), args)
        # env_drift = jnp.array([
        #         jnp.where(jnp.abs(pos) < 5.0, drift[0], 0.0),
        #         jnp.where(jnp.abs(vel) < 1.0, drift[1], 0.0)
        #     ])
            
        return agent_drift, env_drift

    def diffusion(self, t, x, args):
        pytree=(self.agent.diffusion(t, x, args), self.env.diffusion(t, x, args))
        return MixedLinearOperator(pytree, input_structure=self.noise_shape)

    @property
    def noise_shape(self):
        return (self.agent.noise_shape, self.env.noise_shape)