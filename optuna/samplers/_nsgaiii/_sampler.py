from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from optuna._experimental import experimental_class
from optuna.distributions import BaseDistribution
from optuna.samplers._ga import BaseGASampler
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._nsgaiii._elite_population_selection_strategy import (
    NSGAIIIElitePopulationSelectionStrategy,
)
from optuna.samplers._random import RandomSampler
from optuna.samplers.nsgaii._after_trial_strategy import NSGAIIAfterTrialStrategy
from optuna.samplers.nsgaii._child_generation_strategy import NSGAIIChildGenerationStrategy
from optuna.samplers.nsgaii._crossovers._base import BaseCrossover
from optuna.samplers.nsgaii._crossovers._uniform import UniformCrossover
from optuna.search_space import IntersectionSearchSpace
from optuna.trial import FrozenTrial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


@experimental_class("3.2.0")
class NSGAIIISampler(BaseGASampler):
    """Multi-objective sampler using the NSGA-III algorithm.

    NSGA-III stands for "Nondominated Sorting Genetic Algorithm III",
    which is a modified version of NSGA-II for many objective optimization problem.

    For further information about NSGA-III, please refer to the following papers:

    - `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
      Nondominated Sorting Approach, Part I: Solving Problems With Box Constraints
      <https://doi.org/10.1109/TEVC.2013.2281535>`__
    - `An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based
      Nondominated Sorting Approach, Part II: Handling Constraints and Extending to an Adaptive
      Approach
      <https://doi.org/10.1109/TEVC.2013.2281534>`__

    Args:
        reference_points:
            A 2 dimension ``numpy.ndarray`` with objective dimension columns. Represents
            a list of reference points which is used to determine who to survive.
            After non-dominated sort, who out of borderline front are going to survived is
            determined according to how sparse the closest reference point of each individual is.
            In the default setting the algorithm uses `uniformly` spread points to diversify the
            result. It is also possible to reflect your `preferences` by giving an arbitrary set of
            `target` points since the algorithm prioritizes individuals around reference points.

        dividing_parameter:
            A parameter to determine the density of default reference points. This parameter
            determines how many divisions are made between reference points on each axis. The
            smaller this value is, the less reference points you have. The default value is 3.
            Note that this parameter is not used when ``reference_points`` is not :obj:`None`.

    .. note::
        Other parameters than ``reference_points`` and ``dividing_parameter`` are the same as
        :class:`~optuna.samplers.NSGAIISampler`.

    """

    def __init__(
        self,
        *,
        population_size: int = 50,
        mutation_prob: float | None = None,
        crossover: BaseCrossover | None = None,
        crossover_prob: float = 0.9,
        swapping_prob: float = 0.5,
        seed: int | None = None,
        constraints_func: Callable[[FrozenTrial], Sequence[float]] | None = None,
        reference_points: np.ndarray | None = None,
        dividing_parameter: int = 3,
        elite_population_selection_strategy: (
            Callable[[Study, list[FrozenTrial]], list[FrozenTrial]] | None
        ) = None,
        child_generation_strategy: (
            Callable[[Study, dict[str, BaseDistribution], list[FrozenTrial]], dict[str, Any]]
            | None
        ) = None,
        after_trial_strategy: (
            Callable[[Study, FrozenTrial, TrialState, Sequence[float] | None], None] | None
        ) = None,
    ) -> None:
        # TODO(ohta): Reconsider the default value of each parameter.

        if population_size < 2:
            raise ValueError("`population_size` must be greater than or equal to 2.")

        if crossover is None:
            crossover = UniformCrossover(swapping_prob)

        if not isinstance(crossover, BaseCrossover):
            raise ValueError(
                f"'{crossover}' is not a valid crossover."
                " For valid crossovers see"
                " https://optuna.readthedocs.io/en/stable/reference/samplers.html."
            )

        if population_size < crossover.n_parents:
            raise ValueError(
                f"Using {crossover},"
                f" the population size should be greater than or equal to {crossover.n_parents}."
                f" The specified `population_size` is {population_size}."
            )

        super().__init__(population_size=population_size)
        self._random_sampler = RandomSampler(seed=seed)
        self._rng = LazyRandomState(seed)
        self._constraints_func = constraints_func
        self._search_space = IntersectionSearchSpace()

        self._elite_population_selection_strategy = (
            elite_population_selection_strategy
            or NSGAIIIElitePopulationSelectionStrategy(
                population_size=population_size,
                constraints_func=constraints_func,
                reference_points=reference_points,
                dividing_parameter=dividing_parameter,
                rng=self._rng,
            )
        )
        self._child_generation_strategy = (
            child_generation_strategy
            or NSGAIIChildGenerationStrategy(
                crossover_prob=crossover_prob,
                mutation_prob=mutation_prob,
                swapping_prob=swapping_prob,
                crossover=crossover,
                constraints_func=constraints_func,
                rng=self._rng,
            )
        )
        self._after_trial_strategy = after_trial_strategy or NSGAIIAfterTrialStrategy(
            constraints_func=constraints_func
        )

    def reseed_rng(self) -> None:
        self._random_sampler.reseed_rng()
        self._rng.rng.seed()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> dict[str, BaseDistribution]:
        search_space: dict[str, BaseDistribution] = {}
        for name, distribution in self._search_space.calculate(study).items():
            if distribution.single():
                # The `untransform` method of `optuna._transform._SearchSpaceTransform`
                # does not assume a single value,
                # so single value objects are not sampled with the `sample_relative` method,
                # but with the `sample_independent` method.
                continue
            search_space[name] = distribution
        return search_space

    def select_parent(self, study: Study, generation: int) -> list[FrozenTrial]:
        return self._elite_population_selection_strategy(
            study,
            self.get_population(study, generation - 1)
            + self.get_parent_population(study, generation - 1),
        )

    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: dict[str, BaseDistribution],
    ) -> dict[str, Any]:
        generation = self.get_trial_generation(study, trial)
        parent_population = self.get_parent_population(study, generation)
        if len(parent_population) == 0:
            return {}
        return self._child_generation_strategy(study, search_space, parent_population)

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        # Following parameters are randomly sampled here.
        # 1. A parameter in the initial population/first generation.
        # 2. A parameter to mutate.
        # 3. A parameter excluded from the intersection search space.

        return self._random_sampler.sample_independent(
            study, trial, param_name, param_distribution
        )

    def before_trial(self, study: Study, trial: FrozenTrial) -> None:
        self._random_sampler.before_trial(study, trial)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        assert state in [TrialState.COMPLETE, TrialState.FAIL, TrialState.PRUNED]
        self._after_trial_strategy(study, trial, state, values)
        self._random_sampler.after_trial(study, trial, state, values)
