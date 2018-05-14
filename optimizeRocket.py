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
    def __init__(self, *args, **kwargs):
        self._rocket_simulation = nBody.rocket_system(*args, **kwargs)

    def __call__(self, individual_list):
        

def main():

    pop_size = 50

    #initialize ephemerides
    file_name = "../Data/solarSystem.txt"
    eph_data = ephemerides.LimitObjects(file_name, range(10, 2000))

    #initialize rocket evolution w/ eval func
    fitness_func = rocket_eval(rocket_count = pop_size)

    #initialize rocket population
    evolution_environment = evolve.basic_evolution(
        eval_func = fitness_func,
        individual_class = rocket_indiv,
        pop_size = pop_size)
    evolution_environment
    evolution_environment
    evolution_environment
    evolution_environment

    #run simulation
    #print stats
    #show best path

if __name__ == "__main__":
    main()
