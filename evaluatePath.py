import numpy as np

class body_value(object):
    def __init__(self, body = 599, value = 100):
        self.body = body
        self.value = value

    def get_distances(self, rocket_locs, body_locs):
        body_locs = np.expand_dims(body_locs, axis = 1)
        distances = np.linalg.norm(rocket_locs - body_locs, axis = -1)
        return distances

    def __call__(self, times, rocket_locs, body_locs):
        distances = self.get_distance(rocket_locs, body_locs)
        total = np.sum( np.expand_dims(times, 1) * distances, axis = 0 )
        reciprocal = self.value / total
        return reciprocal

class distance_threshold(body_value):
    def __init__(self, distance = 1e8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distance = distance

    def __call__(self, times, rocket_locs, body_locs):
        distances = self.get_distance(rocket_locs, body_locs)
        in_radius = distances < self.distance
        total_time_in = np.sum( np.expand_dims(times, 1) * in_radius, axis = 0 )
        return self.value * total_time_in

class path_evaluator(object):
    def __init__(self, ephemerides, value_list = None):
        self.ephemerides = ephemerides
        self.value_list = value_list
    
    def __call__(self, times, locations):
        scores = np.zeros(shape = locations.shape[1])

        body_set = list(set(value_check.body for value_check in self.value_list))

        body_locs = {body : locs for body, locs in 
            zip(body_set, self.ephemerides.object_paths(objects = body_set, times = times)) }

        for value_check in self.value_list:
            scores += value_check(times, locations,body_locs[value_check.body])

        return scores
            
