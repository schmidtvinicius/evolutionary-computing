import string as string_module
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

    return selections[first_best]


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


def calculate_hamming_distance(string1, string2):
    if len(string1) != len(string2):
        raise RuntimeError('strings should be the same length')
    
    return sum(c1 != c2 for c1, c2 in zip(string1, string2))


def calcualte_population_diversity(population: np.ndarray) -> float:
    diversity = 0
    divisor = 0

    for i in np.arange(len(population)):
        for j in np.arange(i, len(population)):
            divisor += 1
            diversity += calculate_hamming_distance(population[i], population[j])

    return diversity/divisor


# Visualize the distributions of tfinish for each µ (I would suggest a so-called beeswarm plot).
def visualize_distributions(mu, n_gens):
    data = {'mu': mu, 'n_gens': n_gens}
    df = pd.DataFrame(data)

    sns.swarmplot(x='mu', y='n_gens', data=df)
    plt.show()


def run_simulation(
        calculate_diversity=False,
        population_size = 2000,
        word_size = 15,
        mutation_rate = 1/15,
        k = 2
    ):

    for _ in range(11):

        target = generate_target_string(length=word_size)
        population = np.array([generate_target_string(length=word_size) for _ in range(population_size)])
        number_of_generations = 0
        found_target = False

        final_gen_num = []

        while not found_target:
            if calculate_diversity and number_of_generations % 10 == 0:
                print(f'mean Hamming distance at generation {number_of_generations} is {calcualte_population_diversity(population)}')

            number_of_generations += 1
            new_population = np.array([])
            for _ in np.arange(population_size/2):
                parent_0 = tournament_selection(population, target, k)
                parent_1 = tournament_selection(population, target, k)
                
                new_children_0, new_children_1 = crossover(parent_0, parent_1)
                
                new_children_0 = do_mutation(new_children_0, mutation_rate)
                new_children_1 = do_mutation(new_children_1, mutation_rate)
                
                if new_children_0 == target or new_children_1 == target:
                    print(f'found target {target} in generation {number_of_generations}')
                    found_target = True
                    break

                new_population = np.append(new_population, [new_children_0, new_children_1])
            population = new_population
            if number_of_generations > 100:
                print(f'could not find target {target} in 100 generations')
                break


        final_gen_num.append(number_of_generations)

    return np.mean(final_gen_num)

def main():
    mu = [0, 1/15, 3/15]
    n_gens = []
    for i in mu:
        print(f"experiment for mu = {i} ")
        n_gen = run_simulation(True, mutation_rate=i)
        n_gens.append(n_gen)
    visualize_distributions(mu, n_gens)



if __name__ == '__main__':
    main()
