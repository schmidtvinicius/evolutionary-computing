import numpy as np
import random
from scipy.spatial import distance

# Read TSP data from file
def read_tsp(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
         # Read each line from the file and convert each line to a list of floating-point numbers
        coordinates = [[float(val) for val in line.strip().split()] for line in lines]
    return np.array(coordinates)

# Define distance matrix
def calculate_distance_matrix(coordinates):
     # Calculate the pairwise distances between all coordinates using Euclidean distance
    return distance.cdist(coordinates, coordinates, 'euclidean')

# Define random tour generation
def generate_random_tour(num_cities):
    # Generate a random permutation of integers from 0 to num_cities - 1
    return np.random.permutation(num_cities)

# Simple Evolutionary Algorithm (EA)
def simple_ea(num_cities, population_size, generations, dist_matrix):
    # Generate an initial population of random tours
    population = [generate_random_tour(num_cities) for _ in range(population_size)]
    
     # Iterate through a number of generations
    for _ in range(generations):
        # Sort the population based on the length of each tour
        population.sort(key=lambda x: tour_length(x, dist_matrix))
        new_population = []
        
         # Generate a new population through crossover and mutation
        for _ in range(population_size):
            parent1, parent2 = random.sample(population[:population_size // 2], 2)
            child = crossover(parent1, parent2)
            mutation_probability = 0.001
            mutate(child, mutation_probability)
            new_population.append(child)
        
        population = new_population
    
    # Return best tour found in final population
    return min(population, key=lambda x: tour_length(x, dist_matrix))

# Memetic Algorithm (MA) with 2-opt local search
def memetic_algorithm(num_cities, population_size, generations, dist_matrix):
    # Generate an initial population of random tours
    population = [generate_random_tour(num_cities) for _ in range(population_size)]
    
    # Iterate through a number of generations
    for gen in range(generations):
        # Sort the population based on the length of each tour
        population.sort(key=lambda x: tour_length(x, dist_matrix))
        new_population = []
        
         # Apply 2-opt local search to each tour in the population
        for tour in population:
            child = two_opt(tour, dist_matrix)
            new_population.append(child)
        
        population = new_population
        print("GENERETION MA: ", gen)
    
    # Return best tour found in final population
    return min(population, key=lambda x: tour_length(x, dist_matrix))

# Calculate tour length based on distance matrix
def tour_length(tour, dist_matrix):
    return dist_matrix[tour[:-1], tour[1:]].sum() + dist_matrix[tour[-1], tour[0]]

# Crossover operator (OX1)
def crossover(parent1, parent2):
    size = len(parent1)
    # Select two random indices to define a crossover segment
    a, b = random.sample(range(size), 2)
    start, end = min(a, b), max(a, b)

    child = [-1] * size
    # Copy the crossover segment from parent1 to the child
    child[start:end] = parent1[start:end]

    # Fill the remaining positions in the child with elements from parent2
    for elem in parent2:
        if elem not in child[start:end]:
            for i in range(size):
                if child[i] == -1:
                    child[i] = elem
                    break

    return np.array(child)

# Mutation operator (Swap the elements at the selected indices)
def mutate(child, mutation_probability):
    size = len(child)
    for i in range(size):
        if random.random() < mutation_probability:
            # Select another random position to swap with
            swap_index = random.randint(0, size - 1)
            child[i], child[swap_index] = child[swap_index], child[i]

# 2-opt local search
def two_opt(tour, dist_matrix):
    size = len(tour)
    best_tour = tour
    improved = True

    # Continue until no improvement is possible
    while improved:
        improved = False
        for i in range(1, size - 2):
            for j in range(i + 1, size):
                if j - i == 1:
                    continue
                # Reverse the segment between indices i and j
                new_tour = tour[:]
                new_tour[i:j] = tour[j - 1:i - 1:-1]
                # Check if new tour is better than current best tour
                if tour_length(new_tour, dist_matrix) < tour_length(best_tour, dist_matrix):
                    best_tour = new_tour
                    improved = True
        tour = best_tour

    return best_tour

# Main function to compare EA and MA
def compare_algorithms(data, population_size, generations):
    # Read data and calculate matrix
    coordinates = read_tsp(data)
    num_cities = len(coordinates)
    dist_matrix = calculate_distance_matrix(coordinates)
    
    ea_lengths = []
    ma_lengths = []
    repetition = 10
    
    # Repeat experiment
    for i in range(repetition):
        # Run both algorithms and remember tour lengths
        ea_tour = simple_ea(num_cities, population_size, generations, dist_matrix)
        ma_tour = memetic_algorithm(num_cities, population_size, generations, dist_matrix)
        print("----------------------------- i: ", i, "----------------------------")
        
        ea_lengths.append(tour_length(ea_tour, dist_matrix))
        ma_lengths.append(tour_length(ma_tour, dist_matrix))
    
    # Calculate average tour lenghts
    ea_avg_length = sum(ea_lengths) / repetition
    ma_avg_length = sum(ma_lengths) / repetition
    
    print("Simple EA average tour length:", ea_avg_length)
    print("Memetic Algorithm average tour length:", ma_avg_length)

#CALL EVERYTHING
compare_algorithms("/Users/juliabijl/Desktop/Radboud/1st.year/Semester.2/Natural.Computing/file-tsp.txt", population_size=50, generations=1500)
