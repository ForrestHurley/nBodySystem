import numpy as np
import random

class individual(object):
    def __init__(self, data = None, data_shape = (1,)):
        self.score = None
        if data is None:
            self.data_shape = data_shape
        else:
            self.data_shape = data.shape[1:]
        self.data = data

    @classmethod
    def random(cls):
        new_indiv = cls(value)
        new_indiv.add_random_column()
        return new_indiv

    def generate_random_column(self):
        return np.random.rand(self.data_shape)

    def mutate_vals(self, std_dev = 1, proportion = 0.1):
        changes = np.random.rand(len(self.data)) < proportion
        deltas = ( np.random.standard_normal(size=self.data_shape)*std_dev for elem in self.data )
        
        self.data = [elem + delta if change else elem for elem, delta, change in
            zip(self.data, deltas, changes)]

    def add_random_column(self):
        random_col = self.generate_random_column()
        self.data.append(random_col)

    def remove_random_column(self, n = 1):
        if len(self.data) > 0:
            remove_column = np.random.choice(len(self.data),n)
            for rem in remove_column:
                del self.data[rem]

    def mate(self, other):
        rand_self = np.random.rand(len(self.data)) < 0.5
        rand_other = np.random.rand(len(other.data)) < 0.5

        new_self = [ slf for slf, take in zip(self.data, rand_self) if take ]
        new_other = [ oth for oth, take in zip(other.data, rand_other) if take ]

        return random.shuffle(new_self + new_other)

    def __add__(self,x):
        return self.mate(x)

class basic_evaluation(object):
    def __call__(self, indiv_list):
        return [np.sum(indiv) for indiv in indiv_list]

class basic_evolution(object):
    def __init__(self, eval_func = None):
        self.reset_evolution()
        if eval_func is None:
            self.eval_func = basic_evaluation()
        else:
            self.eval_func = eval_func

    def reset_evolution(self):
        self.indiv_list = []
        self.score_list = []
        self.total_generations = 0

    def run_evolution(self, generations = 100, pop_size = 100):
        if self.indiv_list == []:
            self.initialize_pop(size = pop_size)

        for i in range(generations):

            self.score_list.append([
                self.best_score(),
                self.average_score()])

            self.preserve_random_best()
            self.mate_random()
            self.mutate_random()

            self.total_generations += 1

        return self.score_list[-1]

    def get_pop_scores(self):
        need_scores = ((idx, indiv) for idx, indiv in
            zip(range(len(self.indiv_list)),self.indiv_list)
            if indiv.score is None)

        ids, indivs = zip(*need_scores)

        scores = self.eval_func(indivs)
        
        for indiv_id, indiv_score in zip(ids, scores):
            self.indiv_list[indiv_id].score = indiv_score

        out_scores = [indiv.score for indiv in self.indiv_list]
        return out_scores

    def get_sorted_scores(self):
        score_ids = zip(self.get_pop_scores(),range(len(self.indiv_list))
        sorted_scores = sorted(score_ids, key = lambda x : x[0], reverse = True)
        return sorted_scores

    def get_best_score(self):
        return max(self.get_pop_scores())

    def get_average_score(self):
        return sum(self.get_pop_scores()) / float(len(self.indiv_list))

    def preserve_n_best(self, p = 0.1):
        sorted_scores = self.get_sorted_scores()
        n = int(p * len(sorted_scores))

        worst_elems = (elem[1] for elem in sorted_scores[n:])

        for elem in worst_elems:
            del self.indiv_list[elem]
       
    #Only currently works for p <= 0.5 
    def preserve_random_best(self, p = 0.1) #, weight = lambda x : x):
        sorted_scores = self.get_sorted_scores()

        #TODO: support non-linear weighting based on scores
        #transformed = [ (weight(elem[0]), elem[1]) for elem in sorted_scores ]
        #total_weights = sum(transformed)

        loc = np.arange(len(sorted_scores)) / float(len(sorted_scores))
        probs = p * loc * 2

        remove = np.random.rand(probs.shape) > probs
        
        worst_elems = (idx for (score, idx), removal in zip(sorted_scores,remove) if removal)

        for elem in worst_elems:
            del self.indiv_list[elem]

    def mutate_random(self, p = 0.1):
        mutate = np.random.rand(len(self.indiv_list)) < p

        for indiv, mut_bool in zip(self.indiv_list, mutate):
            if mut_bool:
                indiv.mutate()

    def add_remove_random_genes(self, p = 0.1):
        add = np.random.rand(len(self.indiv_list)) < p
        remove = np.random.rand(len(self.indiv_list)) < p

        for indiv, add_bool, rem_bool in zip(self.indiv_list, add, remove):
            if add_bool:
                indiv.add_random_column()
            if rem_bool:
                indiv.remove_random_column()

    def mate_random(self, n = 90):
        pairs = []
        for i in range(n):
            pairs.append(random.sample(self.indiv_list,2))

        new_indivs = [ indivA + indivB for (indivA, indivB) in pairs ]

        self.indiv_list += new_indivs

if __name__ == "__main__":
    pass
