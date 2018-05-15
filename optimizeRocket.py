import evolve
from evaluatePath import path_evaluator
import ephemerides
from nBody import rocket_system
import plot
import time

def rocket_indiv(evolve.individual):
    def __init__(self, max_random_v = 10,
            mean_random_v_delta = 0.5,
            initial_time = None,
            max_random_time = 60*60*24*365,
            mean_random_time_delta = 60*60*24*7):
        #rocket data format (time, vx, vy, vz)
        super().__init__(data = None, data_shape = (4,))
        self.max_random_v = max_random_v
        self.mean_random_v_delta = mean_random_v_delta
        if initial_time = None:
            initial_time = float(time.time())
        self.initial_time = initial_time
        self.max_random_time = max_random_time
        self.mean_random_time_delta = mean_random_time_delta

    def generate_random_column(self):
        return np.concatenate(
            ([self.initial_time + np.random.rand(1) * (self.max_random_time - self.initial_time)],
            np.random.rand(3) * self.max_random_v))

    def mutate_col(self, column):
        column[0] += np.random.standard_normal(size = 1) * self.mean_random_time_delta
        column[1:] += np.random.standard_normal(size = 3) * self.mean_random_v_delta
        return column

    def organize_genes(self):
        self.data.sort(key = lambda x : x[0])

def rocket_eval(evolve.basic_evaluation):
    def __init__(self, total_sim_time = 60*60*24*365.25*2, *args, **kwargs):
        self._rocket_simulation = nBody.rocket_system(*args, **kwargs)
        self.sim_time = total_sim_time
        self.path_eval = path_evaluator()

    def __call__(self, individual_list):
        times, locations = self._rocket_simulation.run_simulation(
            rocket_delta_vs = individual_list,
            total_time = sim_time)

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
        h = 60 * 60 * 12)

    #initialize rocket population
    evolution_environment = evolve.basic_evolution(
        eval_func = fitness_func,
        individual_class = rocket_indiv,
        pop_size = pop_size)
    evolution_environment.keep_proportion = 0.4
    evolution_environment.indiv_mutate_proportion = 0.1
    evolution_environment.gene_mutate_proportion = 0.2
    evolution_environment.add_remove_gene_proportion = 0.01

    #run simulation
    evolution_environment.run(evolution(generations = 20, verbose = True))
    
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
