from abc import ABC, abstractmethod

import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx


@dataclass
class BootstrapperModel:
    "Empty dataclass"


@dataclass
class BootstrapperData:
    "Empty dataclass"


class Bootstrapper:
    def __init__(self, model: BootstrapperModel) -> None:
        """Set the model and simulation parameters.

        Args:
            model: The MuJoCo model to use for simulation.
        """
        self.model = model

    def init_data(self) -> BootstrapperData:
        """Initialize bootstrap data."""
        return BootstrapperData()

    def bootstrap(
        self,
        state: mjx.Data,
        data: BootstrapperData,
        rollout: jnp.ndarray,
        rng: jnp.ndarray,
    ) -> tuple[BootstrapperData, jnp.ndarray]:
        """Get bootstrap data."""
        return data, rollout
