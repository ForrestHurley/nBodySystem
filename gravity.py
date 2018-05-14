import numpy as np
from itertools import permutations, product

class gravity(object):
    def __init__(self, masses, max_acceleration = 1e4, verbose = True):
        self.G = 1 #6.6719199e-11 #https://www.nature.com/articles/nature13433
        self.masses = masses
        self.max_acceleration = max_acceleration
        self.verbose = verbose

    def get_acceleration(self, bodies, *args, **kwargs):
        if not bodies.shape[0] == self.masses.shape[0]:
            raise ValueError("Object positions count not equal to number of masses")
        raw_accelerations = self._get_acceleration(bodies = bodies, *args, **kwargs)
        softened = np.minimum(raw_accelerations, self.max_acceleration)

        if self.verbose:
            soft_count = np.sum(np.logical_not(softened == raw_accelerations))
            if soft_count > 0:
                print("Softened {0} components".format(soft_count))

        return softened

    def _get_acceleration(self, bodies):
        pass

class particle_particle(gravity):
    def two_body_acc(self, posA, posB, massA):
            direction = posA - posB 
            distance = np.expand_dims(np.linalg.norm(direction, axis = -1),-1)
            norm_dir = direction / distance
            raw_acc = (self.G * massA / (distance * distance)) * norm_dir
            if np.all(np.isfinite(raw_acc)):
                return raw_acc
            else:
                return 0
        
    def _get_acceleration(self, bodies):
        body_iteration = permutations(zip(self.masses, bodies, range(self.masses.shape[0])),2)

        accelerations = np.zeros(bodies.shape)

        for (massA, positionA, idxA), (massB, positionB, idxB) in body_iteration:
            accelerations[idxB] += self.two_body_acc(positionA, positionB, massA)

        return accelerations

class particle_rail(particle_particle):
    def _get_acceleration(self, bodies, particles):
        body_rail_iteration = np.array(list(product(range(bodies.shape[0]), range(particles.shape[0]))))

        masses = np.expand_dims(self.masses[body_rail_iteration[:,0]],-1)
        big_pos = bodies[body_rail_iteration[:,0]]
        small_pos = particles[body_rail_iteration[:,1]]

        raw_accelerations = self.two_body_acc(big_pos, small_pos, masses)
        accelerations = np.zeros(particles.shape)
        
        for raw_acc, idx in zip(raw_accelerations, body_rail_iteration[:,1]):
            accelerations[idx] += raw_acc

        return accelerations

class quadropole_tree(gravity):
    def _get_acceleration(self, positions):
        pass

if __name__ == "__main__":
    test_gravity = particle_particle(np.array([10,0,0,0]))
    print(test_gravity.get_acceleration(np.array([[0,0,0],[1,1,1],[10,0,0],[20,0,0]])))
