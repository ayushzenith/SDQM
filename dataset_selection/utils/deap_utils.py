import pickle
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from deap import tools
from deap.algorithms import varAnd
from tqdm import tqdm


def initialize_overlap_individual(
    container: Type, n_1: int, n_2: int, k_l: int, k_u: int
) -> List[int]:
    """Create a list with n_1 + n_2 elements. A random number of
    elements between k_l and k_u will have a value of 1, another
    random number of elements between k_l and k_u will have a value of 2,
    and the rest will have a value of 0.

    :param container: The type to put in the data.
    :param n_1: The number of 1 elements in the list.
    :param n_2: The number of 2 elements in the list.
    :param k_l: The lower bound for the number of 1 and 2 values.
    :param k_u: The upper bound for the number of 1 and 2 values.
    :param seed: Optional seed for reproducibility.
    :returns: An instance of the container filled with the specified data.
    """
    while True:
        size1 = random.randint(k_l, k_u)
        size2 = random.randint(k_l, k_u)
        if size1 + size2 <= n_1 + n_2:
            break

    ind = [1] * size1 + [2] * size2 + [0] * (n_1 + n_2 - size1 - size2)
    random.shuffle(ind)
    return container(ind)


def initialize_no_overlap_individual(
    container: Type, n_1: int, n_2: int, k_l: int, k_u: int
) -> List[bool]:
    """Create a list with n_1 + n_2 elements. The first n_1 elements will have
    a random number of True values between k_l and k_u, and the rest will be False.
    The last n_2 elements will have a random number of True values between k_l and k_u,
    and the rest will be False.

    :param container: The type to put in the data.
    :param n_1: The number of elements in the first part of the list.
    :param n_2: The number of elements in the second part of the list.
    :param k_l: The lower bound for the number of True values in each part.
    :param k_u: The upper bound for the number of True values in each part.
    :returns: An instance of the container filled with the specified data.

    Example:
        >>> initialize_no_overlap_individual(list, 3, 3, 1, 2)
        [True, False, True, True, False, False]
    """
    while True:
        size1 = random.randint(k_l, k_u)
        size2 = random.randint(k_l, k_u)
        if size1 <= n_1 and size2 <= n_2:
            break
    part_1 = [True] * size1 + [False] * (n_1 - size1)
    part_2 = [True] * size2 + [False] * (n_2 - size2)

    # Shuffle the parts to ensure no overlap
    random.shuffle(part_1)
    random.shuffle(part_2)

    return container(part_1 + part_2)


def eaSimple_ckpt(
    population: List[Any],
    toolbox: Any,
    cxpb: float,
    mutpb: float,
    ngen: int,
    values: List[float],
    stats: Optional[tools.Statistics] = None,
    verbose: bool = __debug__,
    save_dir: Optional[str] = None,
    checkpoint: Optional[str] = None,
    min_fitness_threshold: Optional[float] = None,
) -> Tuple[List[Any], tools.Logbook, tools.HallOfFame]:
    """This algorithm reproduce the simplest evolutionary algorithm as
    presented in chapter 7 of [Back2000]_.

    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param cxpb: The probability of mating two individuals.
    :param mutpb: The probability of mutating an individual.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :param checkpoint: A function that will be called every generation with
    :param save_dir: Directory to save checkpoints.
    :param min_fitness_threshold: Minimum fitness value to stop the evolution.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution

    The algorithm takes in a population and evolves it in place using the
    :meth:`varAnd` method. It returns the optimized population and a
    :class:`~deap.tools.Logbook` with the statistics of the evolution. The
    logbook will contain the generation number, the number of evaluations for
    each generation and the statistics if a :class:`~deap.tools.Statistics` is
    given as argument. The *cxpb* and *mutpb* arguments are passed to the
    :func:`varAnd` function. The pseudocode goes as follow ::

        evaluate(population)
        for g in range(ngen):
            population = select(population, len(population))
            offspring = varAnd(population, toolbox, cxpb, mutpb)
            evaluate(offspring)
            population = offspring

    As stated in the pseudocode above, the algorithm goes as follow. First, it
    evaluates the individuals with an invalid fitness. Second, it enters the
    generational loop where the selection procedure is applied to entirely
    replace the parental population. The 1:1 replacement ratio of this
    algorithm **requires** the selection procedure to be stochastic and to
    select multiple times the same individual, for example,
    :func:`~deap.tools.selTournament` and :func:`~deap.tools.selRoulette`.
    Third, it applies the :func:`varAnd` function to produce the next
    generation population. Fourth, it evaluates the new individuals and
    compute the statistics on this population. Finally, when *ngen*
    generations are done, the algorithm returns a tuple with the final
    population and a :class:`~deap.tools.Logbook` of the evolution.

    .. note::

        Using a non-stochastic selection method will result in no selection as
        the operator selects *n* individuals from a pool of *n*.

    This function expects the :meth:`toolbox.mate`, :meth:`toolbox.mutate`,
    :meth:`toolbox.select` and :meth:`toolbox.evaluate` aliases to be
    registered in the toolbox.

    .. [Back2000] Back, Fogel and Michalewicz, "Evolutionary Computation 1 :
       Basic Algorithms and Operators", 2000.
    """
    if checkpoint is not None:
        with open(checkpoint, "rb") as cp:
            cp = pickle.load(cp)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffames = cp["halloffames"]
        values = cp["values"]
        current_hof = cp["current_hof"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        cxpb, mutpb = cp["parameters"]
        toolbox = cp["toolbox"]
        # Update the number of generation to run
        ngen = start_gen + ngen
    else:
        halloffames = [tools.HallOfFame(10)]
        current_hof = 0
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
        start_gen = 0

    # Begin the generational process
    if verbose:
        gen_range = tqdm(range(start_gen + 1, ngen + 1), desc="Generations")
    else:
        gen_range = range(start_gen + 1, ngen + 1)

    for v in range(len(values)):
        # Evaluate all individuals with the new value
        for ind in population:
            ind.fitness.values = toolbox.evaluate(ind, values=values)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        if len(halloffames) <= current_hof:
            halloffames.append(tools.HallOfFame(10))

        halloffames[current_hof].update(population)

        for gen in gen_range:
            elitism_N = 2
            elites = tools.selBest(population, elitism_N)
            elites = [toolbox.clone(ind) for ind in elites]

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - elitism_N)

            # Vary the pool of individuals
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)

            # Add the elites
            offspring.extend(elites)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = [toolbox.evaluate(ind, values=values) for ind in invalid_ind]
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Update the hall of fame with the generated individuals
            halloffames[current_hof].update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print(logbook.stream)

            # save checkpoint
            if save_dir is not None:
                cp = dict(
                    population=population,
                    generation=gen,
                    halloffames=halloffames,
                    current_hof=current_hof,
                    values=values,
                    logbook=logbook,
                    rndstate=random.getstate(),
                    parameters=(cxpb, mutpb),
                    toolbox=toolbox,
                )
                with open(
                    f"{save_dir}/checkpoint_hof{current_hof}_gen{gen}.pkl", "wb"
                ) as cp_file:
                    pickle.dump(cp, cp_file)

            # Early stopping based on minimum fitness threshold
            if min_fitness_threshold is not None:
                min_fitness_index = min(
                    range(len(population)),
                    key=lambda i: population[i].fitness.values[0],
                )
                min_fitness = population[min_fitness_index].fitness.values[0]
                if len(values) > 1:
                    coresponding_value = toolbox.evaluate(
                        population[min_fitness_index],
                        values=values,
                        return_closest=True,
                    )
                    # remove the value from the list
                    values = list(values)
                    values.remove(coresponding_value)
                if min_fitness < min_fitness_threshold:
                    print(
                        f"Early stopping at generation {gen} with min fitness {min_fitness}"
                    )
                    current_hof += 1
                    break

    return population, logbook, halloffames


def mutSwpUniformInt(
    individual: List[int], low: List[int], up: List[int], swappb: float
) -> Tuple[List[int]]:
    """
    Perform a mutation on an individual by swapping elements and replacing elements with random integers.

    This mutation operator performs two types of mutations on the individual:
    1. Swapping two elements with a probability proportional to `swappb / 10000`.
    2. Replacing an element with a random integer within the specified bounds with a probability of `swappb`.

    :param individual: A list representing the individual to be mutated.
    :param low: A list of lower bounds for each element in the individual.
    :param up: A list of upper bounds for each element in the individual.
    :param swappb: The probability of performing a swap or replacement mutation on each element.
    :returns: A tuple containing the mutated individual.

    Example:
        >>> individual = [1, 2, 3, 4, 5]
        >>> low = [0, 0, 0, 0, 0]
        >>> up = [10, 10, 10, 10, 10]
        >>> swappb = 0.1
        >>> mutSwpUniformInt(individual, low, up, swappb)
        ([1, 2, 3, 4, 5],)
    """
    size = len(individual)

    flippb = swappb / 10000

    for i in range(size):
        if random.random() < flippb:
            idx1, idx2 = random.sample(range(size), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        if random.random() < swappb:
            individual[i] = random.randint(low, up)

    return (individual,)
