import numpy as np
import random
import sys
import time

class individual(object):
    def __init__(self, data = None, data_shape = (1,)):
        self.score = None
        if data is None:
            self.data_shape = data_shape
            self.data = []
        elif len(data) < 1:
            self.data_shape = data_shape
            self.data = data
        else:
            self.data_shape = data[0].shape
            self.data = data
        self.std_dev = 1

    @classmethod
    def random(cls, *args, **kwargs):
        new_indiv = cls(*args, **kwargs)
        new_indiv.add_random_column()
        return new_indiv

    def __str__(self):
        return "Genes: {}".format(str(self.data))

    def generate_random_column(self):
        return np.random.rand(*self.data_shape)

    def mutate_vals(self, p = 0.1):

        changes = np.random.rand(len(self.data)) < p
        muted_columns = ( self.mutate_col(elem) for elem in self.data )
        
        self.data = [mut if change else elem for elem, mut, change in
            zip(self.data, muted_columns, changes)]

    def mutate_col(self, column):
        return column + np.random.standard_normal(size = self.data_shape)*self.std_dev

    def add_random_column(self):
        random_col = self.generate_random_column()
        self.data.append(random_col)

    def remove_random_column(self, n = 1):
        if len(self.data) > 0:
            remove_column = np.random.choice(len(self.data),n)
            for rem in sorted(remove_column, reverse = True):
                del self.data[rem]

    def mate(self, other):
        new_gene_count = int(len(self.data) + len(other.data)/2)
        rand_self = list(zip(np.random.rand(len(self.data)), self.data))
        rand_other = list(zip(np.random.rand(len(other.data)), other.data))

        combined = rand_self + rand_other
        sorted(combined)

        child_data = [val[1] for val in combined[:new_gene_count]]
        child = type(self)(data = child_data, data_shape = self.data_shape)
        child.organize_genes()
        return child

    def organize_genes(self):
        random.shuffle(self.data)

    def __add__(self,x):
        return self.mate(x)

class basic_evaluation(object):
    def __call__(self, indiv_list):
        return [self.evaluation(indiv) for indiv in indiv_list]
    def evaluation(self, individual):
        return sum(np.sum(gene) for gene in individual.data)

class basic_evolution(object):
    def __init__(self, eval_func = None, individual_class = None, pop_size = 100):

        if eval_func is None:
            self.eval_func = basic_evaluation()
        else:
            self.eval_func = eval_func
        if individual_class is None:
            self.indiv = individual
        else:
            self.indiv = individual_class

        self.keep_proportion = 0.4
        self.indiv_mutate_proportion = 0.1
        self.gene_mutate_proportion = 0.3
        self.add_remove_gene_proportion = 0.1
        self.preserve_n = 5
        self.topn = int(pop_size * 0.5)

        self.pop_size = pop_size

        self.reset_evolution()

    def reset_evolution(self):
        self.initialize_pop(size = self.pop_size)
        self.score_list = []
        self.total_generations = 0
        self.best_list = []

    def initialize_pop(self, size = 100):
        self.indiv_list = [ self.indiv.random() for i in range(size) ]

    def run_evolution(self, generations = 100, verbose = False):

        last_time = time.time()

        self.score_list.append([
            self.get_best_score(),
            self.get_average_topn_score(n = self.topn),
            self.get_average_score()])

        self.best_list.append(self.get_best())

        for i in range(generations):
            if verbose:
                #sys.stdout.write("\033[K")
                print("Started generation {0}. Best score: {1:.4E}, Average topn score: {2:.4E}, Time: {3:.3E}".format(
                    self.total_generations, 
                    self.score_list[-1][0],
                    self.score_list[-1][1],
                    time.time() - last_time), end = "\n")
                sys.stdout.flush()

            last_time = time.time()

            self.score_list.append([
                self.get_best_score(),
                self.get_average_topn_score(n = self.topn),
                self.get_average_score()])

            self.best_list.append(self.get_best())

            self.preserve_roulette( p = self.keep_proportion )
            self.mutate_random( p = self.indiv_mutate_proportion, preserve_n = self.preserve_n)
            self.mate_random( n = self.pop_size - len(self.indiv_list) )
            self.add_remove_random_genes( p = self.add_remove_gene_proportion )

            self.total_generations += 1

        if verbose:
            print("")

        return self.score_list[-1]

    def get_pop_scores(self):
        need_scores = [(idx, indiv) for idx, indiv in
            zip(range(len(self.indiv_list)),self.indiv_list)
            if indiv.score is None]

        if len(need_scores) > 0:
            ids, indivs = zip(*need_scores)

            scores = self.eval_func(indivs)
            
            for indiv_id, indiv_score in zip(ids, scores):
                self.indiv_list[indiv_id].score = indiv_score

        out_scores = [indiv.score for indiv in self.indiv_list]
        return out_scores

    def get_sorted_scores(self):
        score_ids = zip(self.get_pop_scores(),range(len(self.indiv_list)))
        sorted_scores = sorted(score_ids, key = lambda x : x[0])
        return sorted_scores

    def get_best_score(self):
        return max(self.get_pop_scores())

    def get_average_score(self):
        mean = sum(self.get_pop_scores()) / float(len(self.indiv_list))
        return mean
   
    def get_average_topn_score(self, n = 50):
        sorted_scores = self.get_sorted_scores()
        best_scores = [ elem[0] for elem in sorted_scores[n:] ]
        return sum(best_scores) / float(len(best_scores))
 
    def preserve_n_best(self, p = 0.1):
        sorted_scores = self.get_sorted_scores()
        n = int(p * len(sorted_scores))

        worst_elems = (elem[1] for elem in sorted_scores[:n])

        for elem in sorted(worst_elems, reverse = True):
            del self.indiv_list[elem]
      
    def get_best(self):
        return self.get_n_best(1)[0][0]

    def get_n_best(self, n = 5):
        sorted_scores = self.get_sorted_scores()

        best_elems = (elem for elem in sorted_scores[-n:])

        out_indivs = []
        for elem in best_elems:
            out_indivs.append([self.indiv_list[elem[1]], elem[0]])

        return out_indivs

    def __str__(self):
        best = self.get_n_best(n = 5)

        text_list = [str(elem[0]) + "\tScore: " + str(elem[1]) for elem in best]
        return "\n".join(text_list)
 
    #Only currently works for p <= 0.5 
    def preserve_random_best(self, p = 0.1): #, weight = lambda x : x):
        sorted_scores = self.get_sorted_scores()

        #TODO: support non-linear weighting based on scores
        #transformed = [ (weight(elem[0]), elem[1]) for elem in sorted_scores ]
        #total_weights = sum(transformed)

        loc = np.arange(len(sorted_scores)) / float(len(sorted_scores))
        probs = p * loc * 2

        remove = np.random.rand(*probs.shape) > probs
        
        if len(self.indiv_list) - np.sum(remove) < 2:
            self.preserve_n_best(p = p)
            return

        worst_elems = (idx for (score, idx), removal in zip(sorted_scores,remove) if removal)

        for elem in sorted(worst_elems, reverse = True):
            del self.indiv_list[elem]

    def preserve_roulette(self, p = 0.1):
        sorted_scores = self.get_sorted_scores()

        n = int(len(sorted_scores) * (p))
        scores, indices = zip(*sorted_scores)

        probs = (scores / np.sum(scores))
        
        keep_indice_locs = np.random.choice(len(indices), size = n, replace = False, p = probs)
        
        to_remove_indices = np.array(indices)[-keep_indice_locs]

        for elem in sorted(to_remove_indices, reverse = True):
            del self.indiv_list[elem]

    def mutate_random(self, p = 0.1, preserve_n = 5):
        to_change_indices = [elem[1] for elem in self.get_sorted_scores()[:preserve_n]]

        mutate = np.random.rand(len(to_change_indices)) < p

        for indiv, mut_bool in zip(to_change_indices, mutate):
            if mut_bool:
                self.indiv_list[indiv].mutate_vals( p = self.gene_mutate_proportion )
                self.indiv_list[indiv].score = None

    def add_remove_random_genes(self, p = 0.1):
        add = np.random.rand(len(self.indiv_list)) < p
        remove = np.random.rand(len(self.indiv_list)) < p

        for indiv, add_bool, rem_bool in zip(self.indiv_list, add, remove):
            if add_bool:
                indiv.add_random_column()
            if rem_bool:
                indiv.remove_random_column()
            if add_bool or rem_bool:
                indiv.score = None


    def mate_random(self, n = 90):
        pairs = []
        for i in range(n):
            pairs.append(random.sample(self.indiv_list,2))

        new_indivs = [ indivA + indivB for (indivA, indivB) in pairs ]

        self.indiv_list += new_indivs

if __name__ == "__main__":
    class test_func(basic_evaluation):
        def evaluation(self, individual):
            if len(individual.data) > 0:
                return np.sum(np.sin(individual.data[0]**2)-np.exp(individual.data[0]**-1)) - len(individual.data)**2 + 10 * len(individual.data) + 1
            else:
                return -10

    eval_func = test_func()
    indiv_class = individual
    evo_environment = basic_evolution(
        eval_func = eval_func,
        individual_class = indiv_class,
        pop_size = 25)
    evo_environment.keep_proportion = 0.4
    evo_environment.indiv_mutate_proportion = 0.5
    evo_environment.gene_mutate_proportion = 1
    evo_environment.add_remove_gene_proportion = 0.002
    
    evo_environment.run_evolution(generations = 20, verbose = True)
    scores = np.array(evo_environment.score_list)

    print(evo_environment)

    from matplotlib import pyplot as plt
    plt.plot(scores[:,0],label="Top Score")
    plt.plot(scores[:,1],label="Average Score of Top 50%")
    plt.legend()
    plt.show()
