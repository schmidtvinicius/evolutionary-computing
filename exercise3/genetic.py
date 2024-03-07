import random
import string as string_module
import numpy as np
import matplotlib.pyplot as plt


def generate_target_string(alphabet=string_module.ascii_letters, length=15) -> str:
    """
    Generate a random string of length `length` using the characters in `alphabet`
    
    Args:
    alphabet (str): the characters to use for the string
    length (int): the length of the string
    
    Returns:
    str: the generated string
    """
    target = ''
    for _ in np.arange(length):
        character_index = np.random.randint(0, len(alphabet))
        target += alphabet[character_index]
    return target


def calculate_fitness(target: str, candidate: str) -> float:
    """
    Calculate the fitness of a candidate string compared to a target string
    
    Args:
    target (str): the target string
    candidate (str): the candidate string
    
    Returns:
    float: the fitness score
    """
    if len(target) != len(candidate):
        raise RuntimeError('strings should be the same length')

    # print(f'comparing target {target} to candidate {candidate}')

    fitness = 0
    for i in range(len(target)):
        if target[i] == candidate[i]:
            fitness += 1
    return fitness / len(target)


def tournament_selection(population: np.ndarray, target: str, k: int) -> list[str]:
    """
    """

    indexes = np.random.choice(len(population), k, replace=False)
    selections = population[indexes]

    selection_fitness = [calculate_fitness(target, selection) for selection in selections]
    
    first_best = np.argmax(selection_fitness)
    first_idx = indexes[first_best]

    return selections[first_best], first_idx

def crossover(parent_0: str, parent_1: str) -> list:
    """
    """
    if len(parent_0) != len(parent_1):
        raise RuntimeError('parents should be the same length')

    crossover_point = np.random.randint(0, len(parent_0))
    new_parent_0 = parent_0[:crossover_point] + parent_1[crossover_point:]
    new_parent_1 = parent_1[:crossover_point] + parent_0[crossover_point:]
    return [new_parent_0, new_parent_1]


def do_mutation(string: str, mu: float) -> str:
    for i in range(len(string)):
        if np.random.rand() < mu:
            idx = np.random.choice(len(string_module.ascii_letters))
            string = string[:i] + string_module.ascii_letters[idx] + string[i+1:]
    return string


def remove_parents(population: list, first_idx: int, second_idx: int) -> list:
    new_pop = []
    for idx, word in enumerate(population):
        if idx != first_idx and idx != second_idx:
            new_pop.append(word)
    return new_pop



def add_children(population: np.ndarray, children: np.ndarray) -> np.ndarray:
    return np.append(population, children)


population_size = 200
word_size = 15
mutation_rate = 1/word_size
k = 2

target = generate_target_string(length=word_size)
population = np.array([generate_target_string(length=word_size) for _ in range(population_size)])

for _ in range(10):
    number_of_generations = 0
    while True:
        number_of_generations += 1
        # print(number_of_generations)
        
        parent_0, first_idx = tournament_selection(population, target, k)
        parent_1, second_idx = tournament_selection(population, target, k)
        while first_idx == second_idx:
            parent_1, second_idx = tournament_selection(population, target, k)
        
        new_children_0, new_children_1 = crossover(parent_0, parent_1)
        
        new_children_0 = do_mutation(new_children_0, mutation_rate)
        new_children_1 = do_mutation(new_children_1, mutation_rate)
        
        population = remove_parents(population, first_idx, second_idx)
        population = add_children(population, [new_children_0, new_children_1])

        if number_of_generations % 100 == 0:
            print(f'generation {number_of_generations}')
            print(f'best candidate: {population[np.argmax([calculate_fitness(target, candidate) for candidate in population])]}')
            print(f'fitness: {np.max([calculate_fitness(target, candidate) for candidate in population])}')
            print(f'average fitness: {np.mean([calculate_fitness(target, candidate) for candidate in population])}')
            print(f'population size: {len(population)}')
        
        if new_children_0 == target or new_children_1 == target:
            print(f'found target {target} in generation {number_of_generations}')
            break
