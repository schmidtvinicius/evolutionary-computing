# Description: Genetic Algorithm
import numpy as np
import matplotlib.pyplot as plt
import time



class GeneticAlgorithm:
    def __init__(self, l, mu, generations, replace_type='closest'):
        self.l = l
        self.mu = mu
        self.generations = generations
        self.replace_type = replace_type
        self.all_fitness = []

    def generate_bit_sequence(self):
        return np.random.randint(0, 2, self.l)
    
    def create_copy(self, x):
        xm = np.copy(x)
        for j in range(self.l):
            if np.random.rand() < self.mu:
                xm[j] = 1 - xm[j]
        return xm
    
    def fitness(self, x):
        return np.sum(x)
    
    def replace(self, x, xm):
        if self.replace_type == 'closest':
            if self.fitness(xm) > self.fitness(x):
                return xm
            else:
                return x
        elif self.replace_type == 'always':
            return xm
        else:
            raise ValueError('Invalid replace type')
    
    def run(self, verbose=False):
        x = self.generate_bit_sequence()
        for i in range(self.generations):
            xm = self.create_copy(x)
            x = self.replace(x, xm)
            current_fitness = self.fitness(x)
            self.all_fitness.append(current_fitness)
            if verbose:
                print(f'Generation: {i}, Fitness: {current_fitness}')
        return self.all_fitness
    
    def plot(self, show=False, save=True, filename='best_fitness.png'):
        plt.plot(self.all_fitness)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness against the elapsed number of generations')
        
        if save:
            plt.savefig(filename)

        if show:
            plt.show()

        plt.close()


if __name__ == "__main__":
    # Parameters
    l = 100
    mu = 1/l
    generations = 1500
    verbose = False

    all_fit_1 = []
    all_fit_2 = []

    for i in range(10):
        if verbose:
            print(f'Run {i+1}')
        
        # Run the GA
        ga1 = GeneticAlgorithm(l, mu, generations, replace_type='closest')
        all_fitness1 = ga1.run()
        all_fit_1.append(all_fitness1)
        ga1.plot(save=True, filename=f'exercise21_run{i+1}.png')

        ga2 = GeneticAlgorithm(l, mu, generations, replace_type='always')
        all_fitness2 = ga2.run()
        all_fit_2.append(all_fitness2)
        ga2.plot(save=True, filename=f'exercise22_run{i+1}.png')

        # Plot the best fitness against the elapsed number of generations in both cases
        plt.plot(all_fitness1, label='Closest')
        plt.plot(all_fitness2, label='Always')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness against the elapsed number of generations')
        plt.legend()
        plt.savefig(f'exercise22_final_run{i+1}.png')
        if verbose:
            plt.show()
        plt.close()

    # Take the average over the columns of the all_fit arrays
    all_fit_1 = np.array(all_fit_1)
    all_fit_2 = np.array(all_fit_2)

    all_fit_1 = np.mean(all_fit_1, axis=0)
    all_fit_2 = np.mean(all_fit_2, axis=0)

    # Plot the avg fitness against the elapsed number of generations
    plt.plot(all_fit_1, label='Closest')
    plt.plot(all_fit_2, label='Always')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Average fitness against the elapsed number of generations over 10 runs')
    plt.legend()
    plt.savefig('exercise23.png')
    if verbose:
        plt.show()
