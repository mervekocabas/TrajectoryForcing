from math import sqrt
from typing import Any
import jax.numpy as jnp
import jax.random as jr
from flax import linen as nn


class TorchLinear(nn.Module):
    """A linear layer similar to torch.nn.Linear."""

    in_features: int
    out_features: int
    bias: bool = True
    weight_init: str = "scaled_variance"
    init_constant: float = 1.0
    bias_init: str = "zeros"  # options: 'zeros'
    dtype: Any = None  # computation dtype (None = input dtype)

    def setup(self):
        """Setup the linear layer with the specified initialization."""

        if self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.in_features)
            weight_initializer = nn.initializers.normal(stddev=std)
        elif self.weight_init == "zeros":
            weight_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        if self.bias_init == "zeros":
            bias_initializer = nn.initializers.zeros
        else:
            raise ValueError(f"Invalid bias_init: {self.bias_init}")

        self._flax_linear = nn.Dense(
            features=self.out_features,
            use_bias=self.bias,
            kernel_init=weight_initializer,
            bias_init=bias_initializer,
            dtype=self.dtype,
        )

    def __call__(self, x):
        return self._flax_linear(x)


class TorchEmbedding(nn.Module):
    """A embedding layer similar to torch.nn.Embedding."""

    num_embeddings: int
    embedding_dim: int
    weight_init: str = "scaled_variance"
    init_constant: float = 1.0

    def setup(self):
        """Setup the embedding layer with the specified initialization."""

        if self.weight_init is None:
            std = 0.02
        elif self.weight_init == "scaled_variance":
            std = self.init_constant / sqrt(self.embedding_dim)
        else:
            raise ValueError(f"Invalid weight_init: {self.weight_init}")

        init = lambda key, shape, dtype: jr.normal(key, shape, dtype) * std
        self._flax_embedding = self.param(
            "embedding", init, (1, self.num_embeddings, self.embedding_dim), dtype=jnp.float32
        )

    def __call__(self, x):
        return self._flax_embedding[:, x, :][0]


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""

    dim: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param("kernel", nn.initializers.ones, (self.dim,))

    def _norm(self, x):
        # Always compute in float32 for numerical stability
        x_f = x.astype(jnp.float32)
        mean_square = jnp.mean(jnp.square(x_f), axis=-1, keepdims=True)
        return x_f * jnp.reciprocal(jnp.sqrt(mean_square + self.eps))

    def __call__(self, x):
        output = self._norm(x)
        return (output * self.weight).astype(x.dtype)


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP."""

    in_features: int
    hidden_features: int

    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0
    dtype: Any = None

    def setup(self):
        init_kwargs = dict(
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
            dtype=self.dtype,
        )

        self.w1 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w3 = TorchLinear(self.in_features, self.hidden_features, **init_kwargs)
        self.w2 = TorchLinear(self.hidden_features, self.in_features, **init_kwargs)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))