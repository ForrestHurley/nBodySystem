import numpy as np
from itertools import permutations

class gravity(object):
    def __init__(self, masses, max_acceleration = 1e4, verbose = True):
        self.G = 1 #6.6719199e-11 #https://www.nature.com/articles/nature13433
        self.masses = masses
        self.max_acceleration = max_acceleration
        self.verbose = verbose

    def get_acceleration(self, positions):
        if not positions.shape[0] == self.masses.shape[0]:
            raise ValueError("Object positions count not equal to number of masses")
        raw_accelerations = self._get_acceleration(positions)
        softened = np.minimum(raw_accelerations, self.max_acceleration)

        if self.verbose:
            soft_count = np.sum(np.logical_not(softened == raw_accelerations))
            if soft_count > 0:
                print("Softened {0} components".format(soft_count))

        return softened

    def _get_acceleration(self, positions):
        pass

class particle_particle(gravity):
    def _get_acceleration(self, positions):
        body_iteration = permutations(zip(self.masses, positions, range(self.masses.shape[0])),2)

        accelerations = np.zeros(positions.shape)

        for (massA, positionA, idxA), (massB, positionB, idxB) in body_iteration:
            direction = positionA - positionB 
            distance = np.linalg.norm(direction, axis = -1)
            norm_dir = direction / distance
            acc = (self.G * massA / (distance * distance)) * norm_dir
            accelerations[idxB] += acc

        return accelerations

class quadropole_tree(gravity):
    def _get_acceleration(self, positions):
        pass

if __name__ == "__main__":
    test_gravity = particle_particle(np.array([10,0,0,0]))
    print(test_gravity.get_acceleration(np.array([[0,0,0],[1,1,1],[10,0,0],[20,0,0]])))
