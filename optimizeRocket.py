import evolve
from evaluatePath import path_evaluator, body_value, distance_threshold
from ephemerides import ephemerides
import nBody
import plot
import time
import numpy as np

class rocket_indiv(evolve.individual):
    def __init__(self, max_random_v = 10,
            mean_random_v_delta = 0.5,
            initial_time = None,
            max_random_time = 60*60*24*365,
            mean_random_time_delta = 60*60*24*7):
        #rocket data format (time, vx, vy, vz)
        super().__init__(data = None, data_shape = (4,))
        self.max_random_v = max_random_v
        self.mean_random_v_delta = mean_random_v_delta
        if initial_time is None:
            initial_time = float(time.time())
        self.initial_time = initial_time
        self.max_random_time = max_random_time
        self.mean_random_time_delta = mean_random_time_delta

    def generate_random_column(self):
        return np.concatenate(
            [self.initial_time + np.random.rand(1) * (self.max_random_time - self.initial_time),
            np.random.rand(3) * self.max_random_v])

    def mutate_col(self, column):
        column[0] += np.random.standard_normal(size = 1) * self.mean_random_time_delta
        column[1:] += np.random.standard_normal(size = 3) * self.mean_random_v_delta
        return column

    def organize_genes(self):
        self.data.sort(key = lambda x : x[0])

class rocket_eval(evolve.basic_evaluation):
    def __init__(self, ephemerides, value_list = None, total_sim_time = 60*60*24*365.25*2, verbose = False, *args, **kwargs):
        self._rocket_simulation = nBody.rocket_system(ephemerides = ephemerides, *args, **kwargs)
        self.sim_time = total_sim_time
        self.verbose = verbose
        
        if value_list is None:
            value_list = [body_value(body = 599, value = 100)]

        self.path_eval = path_evaluator(ephemerides = ephemerides, value_list = value_list)

    def __call__(self, individual_list):
        times, locations = self._rocket_simulation.run_simulation(
            rocket_delta_vs = [rocket.data for rocket in individual_list],
            total_time = self.sim_time,
            verbose = self.verbose)

        rocket_scores = self.path_eval(times, locations)

        return rocket_scores

def main():

    pop_size = 50

    #initialize ephemerides
    file_name = "../Data/solarSystem.txt"
    eph_data = ephemerides.LimitObjects(file_name, range(10, 2000))

    #initialize rocket evolution w/ eval func
    fitness_func = rocket_eval(ephemerides = eph_data,
        rocket_count = pop_size,
        start_time = float(time.time()),
        total_sim_time = 60 * 60 * 24 * 365.25 * 1,
        h = 60 * 60 * 12,
        verbose = False)

    #initialize rocket population
    evolution_environment = evolve.basic_evolution(
        eval_func = fitness_func,
        individual_class = rocket_indiv,
        pop_size = pop_size)
    evolution_environment.keep_proportion = 0.4
    evolution_environment.indiv_mutate_proportion = 0.1
    evolution_environment.gene_mutate_proportion = 0.2
    evolution_environment.add_remove_gene_proportion = 0.1

    #run simulation
    evolution_environment.run_evolution(generations = 20, verbose = True)
    
    #print stats
    print(evolution_environment)

    scores = np.array(evolution_environment.score_list)
    from matplotlib import pyplot as plt
    plt.plot(scores[:,0], label = "Top Score")
    plt.plot(scores[:,1], label = "Average Score of Top 50%")
    plt.legend()
    plt.show()

    #show best path

if __name__ == "__main__":
    main()
