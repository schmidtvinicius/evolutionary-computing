import numpy as np
import random
from scipy.spatial import distance

# Read TSP data from file
def read_tsp(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        coordinates = [[float(val) for val in line.strip().split()] for line in lines]
    return np.array(coordinates)

# Define distance matrix
def calculate_distance_matrix(coordinates):
    return distance.cdist(coordinates, coordinates, 'euclidean')

# Define random tour generation
def generate_random_tour(num_cities):
    return np.random.permutation(num_cities)

# Simple Evolutionary Algorithm (EA)
def simple_ea(num_cities, population_size, generations, dist_matrix):
    population = [generate_random_tour(num_cities) for _ in range(population_size)]
    
    for gen in range(generations):
        population.sort(key=lambda x: tour_length(x, dist_matrix))
        new_population = []
        
        for _ in range(population_size):
            parent1, parent2 = random.sample(population[:population_size // 2], 2)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        
        population = new_population
        print("GENERATION: ", gen)
    
    return min(population, key=lambda x: tour_length(x, dist_matrix))

# Memetic Algorithm (MA) with 2-opt local search
def memetic_algorithm(num_cities, population_size, generations, dist_matrix):
    population = [generate_random_tour(num_cities) for _ in range(population_size)]
    
    for gen in range(generations):
        population.sort(key=lambda x: tour_length(x, dist_matrix))
        new_population = []
        
        for tour in population:
            child = two_opt(tour, dist_matrix)
            new_population.append(child)
        
        population = new_population
        print("GENERETION MA: ", gen)
    
    return min(population, key=lambda x: tour_length(x, dist_matrix))

# Calculate tour length
def tour_length(tour, dist_matrix):
    return dist_matrix[tour[:-1], tour[1:]].sum() + dist_matrix[tour[-1], tour[0]]

# Crossover operator (OX1)
def crossover(parent1, parent2):
    size = len(parent1)
    a, b = random.sample(range(size), 2)
    start, end = min(a, b), max(a, b)

    child = [-1] * size
    child[start:end] = parent1[start:end]

    for elem in parent2:
        if elem not in child[start:end]:
            for i in range(size):
                if child[i] == -1:
                    child[i] = elem
                    break

    return np.array(child)

# Mutation operator (Swap)
def mutate(tour):
    size = len(tour)
    a, b = random.sample(range(size), 2)
    tour[a], tour[b] = tour[b], tour[a]

# 2-opt local search
def two_opt(tour, dist_matrix):
    size = len(tour)
    best_tour = tour
    improved = True

    while improved:
        improved = False
        for i in range(1, size - 2):
            for j in range(i + 1, size):
                if j - i == 1:
                    continue
                new_tour = tour[:]
                new_tour[i:j] = tour[j - 1:i - 1:-1]
                if tour_length(new_tour, dist_matrix) < tour_length(best_tour, dist_matrix):
                    best_tour = new_tour
                    improved = True
        tour = best_tour

    return best_tour

# Main function to compare EA and MA
def compare_algorithms(data, population_size, generations):
    coordinates = read_tsp(data)
    num_cities = len(coordinates)
    dist_matrix = calculate_distance_matrix(coordinates)
    
    ea_lengths = []
    ma_lengths = []
    repetition = 10
    
    for _ in range(repetition):
        ea_tour = simple_ea(num_cities, population_size, generations, dist_matrix)
        ma_tour = memetic_algorithm(num_cities, population_size, generations, dist_matrix)
        
        ea_lengths.append(tour_length(ea_tour, dist_matrix))
        ma_lengths.append(tour_length(ma_tour, dist_matrix))
    
    ea_avg_length = sum(ea_lengths) / repetition
    ma_avg_length = sum(ma_lengths) / repetition
    
    print("Simple EA average tour length:", ea_avg_length)
    print("Memetic Algorithm average tour length:", ma_avg_length)

#CALL EVERYTHING
compare_algorithms("/Users/juliabijl/Desktop/Radboud/1st.year/Semester.2/Natural.Computing/file-tsp.txt", population_size=50, generations=1500)
