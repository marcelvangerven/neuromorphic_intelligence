import jax
import jax.numpy as jnp
import jax.random as jr
import diffrax as dfx
import equinox as eqx
from diffrax._custom_types import RealScalarLike
from lineax._custom_types import sentinel
from lineax._operator import _frozenset
import lineax as lx
from jaxtyping import PyTree, Inexact, Array
import jax.tree_util as jtu

from lineax._tags import (
    diagonal_tag,
    lower_triangular_tag,
    negative_semidefinite_tag,
    positive_semidefinite_tag,
    symmetric_tag,
    transpose_tags,
    tridiagonal_tag,
    unit_diagonal_tag,
    upper_triangular_tag,
)

## dict get

@eqx.filter_jit
def dict_get(d, key, default=None):
    """
    Retrieve the value associated with a given key from a dictionary.
    Args:
        d (dict): The dictionary to search for the key.
        key (str): The key whose value needs to be retrieved.
        default (optional): The value to return if the key is not found. Defaults to None.
    Returns:
        The value associated with the dictionary and/ or key exist, otherwise the default value.
    """
    return default if d is None else d.get(key, default)

## MixedLinearOperator

class MixedLinearOperator(lx.AbstractLinearOperator, strict=True):

    pytree: PyTree
    output_structure: PyTree = eqx.field(static=True)
    tags: frozenset[object] = eqx.field(static=True)
    input_structure: PyTree = eqx.field(static=True)

    def __init__(self, pytree, input_structure=None, output_structure=None, tags = ()):
        """
        Allows for a PyTree of scalars, arrays, and linear operators to be treated as a single linear operator. Either input_structure or output_structure should be provided, and the other will be inferred from the structure of the pytree.

        Leafs can be of the following types:
            - None: this leaf will return the untransformed input vector.
            - RealScalarLike: this leaf will be treated as a scalar multiply.
            - jnp.ndarray: this leaf will be treated as a matrix linear operator.
            - lineax.AbstractLinearOperator: this leaf will be treated as a linear operator.

        Example:
        ```python
        import jax
        import jax.numpy as jnp
        import lineax as lx
        import jax
        import jax.numpy as jnp
        s1 = jax.ShapeDtypeStruct((), dtype=jnp.float32)

        m = MixedLinearOperator((2.0, None, (2.0, 2.0), lx.IdentityLinearOperator(input_structure=s1)), output_structure=(s1, s1, (s1, jax.ShapeDtypeStruct((2,), dtype=jnp.float32)), s1)) 
        print(m.mv((2.0, 2.0, (3.0, 2*jnp.ones((2,1))), 1.0)))

        m = MixedLinearOperator((2.0, None, (2.0, 2.0), lx.IdentityLinearOperator(input_structure=s1)), input_structure=(s1, s1, (s1, jax.ShapeDtypeStruct((1,), dtype=jnp.float32)), s1)) 
        print(m.mv((2.0, 2.0, (3.0, 2*jnp.ones((2,1))), 1.0)))
        ```
        
        **Arguments:**

        - `pytree`: this should be a PyTree, with structure as specified in
            [`lineax.PyTreeLinearOperator`][].
        - `input_structure`: the structure of the input space. This should be a PyTree
        - `output_structure`: the structure of the output space. This should be a PyTree
            of `jax.ShapeDtypeStruct`s. (The structure of the input space is then
            automatically derived from the structure of `pytree`.)
        - `tags`: any tags indicating whether this operator has any particular
            properties, like symmetry or positive-definite-ness. Note that these
            properties are unchecked and you may get incorrect values elsewhere if these
            tags are wrong.
        """
        self.pytree = pytree
        self.tags = _frozenset(tags)

        assert input_structure is not None or output_structure is not None, "Either input_structure or output_structure should be provided."

        if input_structure is None:
    
            self.output_structure = output_structure

            def _input_structure(leaf, out_shape):
                if leaf is None:
                    return out_shape
                elif isinstance(leaf, RealScalarLike):
                    return out_shape
                elif isinstance(leaf, jnp.ndarray):
                    # We have to be careful here since the tensordot operation can act on any tensors op[d1, d2, ..., dn] and vec[d1, d2, ..., dm] and return output [d1, d2, ..., dk] where k = n - m
                    # Since we cannot infer the shape of leaf here, we can only assume that m = n+1. If we want to operator on arbitrary tensors then we should explicitly provide both the input and output shapes.
                    if jnp.ndim(leaf) == 1 or jnp.ndim(leaf) == 0:
                        return jax.ShapeDtypeStruct((leaf.size,), dtype=leaf.dtype)
                    elif jnp.ndim(leaf) == 2:
                        return jax.ShapeDtypeStruct((leaf.shape[0],), dtype=leaf.dtype)
                    else:
                        raise ValueError(f"Unsupported leaf dimension: {jnp.ndim(leaf)}; consider providing input_structure explicitly.")
                elif isinstance(leaf, lx.AbstractLinearOperator):
                    return leaf.in_structure()
                else:
                    raise ValueError(f"Unsupported leaf type: {type(leaf)}")

            self.input_structure = jax.tree.map(_input_structure, pytree, output_structure, is_leaf=lambda x:(isinstance(x, lx.AbstractLinearOperator) or x is None))

        if output_structure is None:

            self.input_structure = input_structure

            def _output_structure(leaf, in_shape):
                if leaf is None:
                    return in_shape
                elif isinstance(leaf, RealScalarLike):
                    return in_shape
                elif isinstance(leaf, jnp.ndarray):
                    if jnp.ndim(leaf) == 1 or jnp.ndim(leaf) == 0:
                        return jax.ShapeDtypeStruct((leaf.size,), dtype=leaf.dtype)
                    elif jnp.ndim(leaf) == 2:
                        return jax.ShapeDtypeStruct((leaf.shape[0],), dtype=leaf.dtype)
                    else:
                        raise ValueError(f"Unsupported leaf dimension: {jnp.ndim(leaf)}; consider providing output_structure explicitly.")
                elif isinstance(leaf, lx.AbstractLinearOperator):
                    return leaf.out_structure()
                else:
                    raise ValueError(f"Unsupported leaf type: {type(leaf)}")

            self.output_structure = jax.tree.map(_output_structure, pytree, input_structure, is_leaf=lambda x: (isinstance(x, lx.AbstractLinearOperator) or x is None))
    
    def mv(self, vector):

        def is_leaf(leaf):
            return (leaf is None or isinstance(leaf, RealScalarLike) or isinstance(leaf, jnp.ndarray) or isinstance(leaf, lx.AbstractLinearOperator))

        def operator(leaf, vector):
            if leaf is None:             
                return 0.0 # forces the output to be zero irrespective of the input
            elif isinstance(leaf, RealScalarLike):
                return leaf * vector
            elif isinstance(leaf, jnp.ndarray):
                # diffrax uses this for noise multiplication. lineax.MatrixLinearOperator uses jnp.matmul(leaf, vector)
                return jnp.tensordot(jnp.conj(leaf), vector, axes=jnp.ndim(vector))
            elif isinstance(leaf, lx.AbstractLinearOperator):
                return leaf.mv(vector)
            else:
                raise ValueError(f"Unsupported leaf type: {type(leaf)}")

        return jax.tree.map(operator, self.pytree, vector, is_leaf=is_leaf)

    def as_matrix(self) -> Inexact[Array, "a b"]:
        raise NotImplementedError

    def transpose(self) -> "AbstractLinearOperator":
        if symmetric_tag in self.tags:
            return self
        raise NotImplementedError

    def in_structure(self):
        return self.input_structure
        # leaves, treedef = self.input_structure
        # return jax.tree.unflatten(treedef, leaves)

    def out_structure(self):
        return self.output_structure
        # leaves, treedef = self.output_structure
        # return jax.tree.unflatten(treedef, leaves)

@lx.linearise.register(MixedLinearOperator)
def _(operator):
    return operator

@lx.materialise.register(MixedLinearOperator)
def _(operator):
    return operator

@lx.diagonal.register(MixedLinearOperator)
def _(operator):
    return jnp.diag(operator.as_matrix())

@lx.tridiagonal.register(MixedLinearOperator)
def _(operator):
    matrix = operator.as_matrix()
    assert matrix.ndim == 2
    diagonal = jnp.diagonal(matrix, offset=0)
    upper_diagonal = jnp.diagonal(matrix, offset=1)
    lower_diagonal = jnp.diagonal(matrix, offset=-1)
    return diagonal, lower_diagonal, upper_diagonal

@lx.is_symmetric.register(MixedLinearOperator)
def _(operator):
    return any(
        tag in operator.tags
        for tag in (
            symmetric_tag,
            positive_semidefinite_tag,
            negative_semidefinite_tag,
            diagonal_tag,
        )
    )

@lx.is_diagonal.register(MixedLinearOperator)
def _(operator):
    return diagonal_tag in operator.tags or (
        operator.in_size() == 1 and operator.out_size() == 1
    )

@lx.is_tridiagonal.register(MixedLinearOperator)
def _(operator):
    return tridiagonal_tag in operator.tags or diagonal_tag in operator.tags

@lx.has_unit_diagonal.register(MixedLinearOperator)
def _(operator):
    return unit_diagonal_tag in operator.tags

@lx.is_lower_triangular.register(MixedLinearOperator)
def _(operator):
    return lower_triangular_tag in operator.tags

@lx.is_upper_triangular.register(MixedLinearOperator)
def _(operator):
    return upper_triangular_tag in operator.tags

@lx.is_positive_semidefinite.register(MixedLinearOperator)
def _(operator):
    return positive_semidefinite_tag in operator.tags

@lx.is_negative_semidefinite.register(MixedLinearOperator)
def _(operator):
    return negative_semidefinite_tag in operator.tags

@lx.conj.register(MixedLinearOperator)
def _(operator):
    pytree_conj = jtu.tree_map(lambda x: x.conj(), operator.pytree)
    return MixedLinearOperator(pytree_conj, operator.out_structure(), operator.tags)
