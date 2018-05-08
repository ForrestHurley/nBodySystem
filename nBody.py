import numpy as np
import integrator
import gravity
import csv

class body_differentials(integrator.differential_equation):
    def __init__(self, masses, *args, **kwargs):
        super().__init__()
        if len(masses.shape) == 1:
            masses = np.expand_dims(masses, axis = 1)
        self.gravity = gravity.particle_particle(masses = masses,*args,**kwargs)

    #state is a nx2x3 array where state[:,0] is positions and state[:,1] is velocities
    def evaluate(self, state, constant_args = (), time = 0):
        loc, vel = system.state_to_loc_vel(state)
        dv = self.gravity.get_acceleration(loc)
        dx = vel
        return system.loc_vel_to_state(dx,dv)

class system(object):
    def __init__(self):
        self.gravity = gravity.particle_particle
        self.integrate = integrator.bulirsch_stoer
        self._locations = None
        self._masses = np.array([])
        self._velocities = None
        self._names = np.array([])
        self._history = []

    @classmethod
    def from_mass_state(cls, masses, locations, velocities):
        new_system = cls()
        new_system.add_bodies(masses, locations, velocities)
        return new_system

    @classmethod
    def from_file(cls, file_name):
        new_system = cls()
        new_system.add_bodies_from_file(file_name)
        return new_system

    @classmethod
    def from_state(cls, state, masses):
        new_system = cls()
        new_system.masses = masses
        new_system.set_state(state)
        return new_system

    def set_state(self, state):
        self._locations = state[:,0]
        self._velocities = state[:,1]

    def get_state(self):
        return np.stack([self._locations, self._velocities], axis = 1)

    @staticmethod
    def state_to_loc_vel(state):
        return state[:,0], state[:,1]

    @staticmethod
    def loc_vel_to_state(loc, vel):
        return np.stack([loc, vel], axis = 1)

    def add_bodies_from_file(self, file_name):
        with open(file_name,'r') as csv_file:
            reader = csv.reader(csv_file)
            headers = reader.__next__()
            loc_list = []
            mass_list = []
            vel_list = []
            name_list = []
            for row in reader:
                name_list.append(row[0])
                mass_list.append(row[1])
                loc_list.append(row[2:5])
                vel_list.append(row[5:8])
            self.add_bodies(
                masses = np.array(mass_list, dtype = 'float'),
                locations = np.array(loc_list, dtype = 'float'),
                velocities = np.array(vel_list, dtype = 'float'),
                names = np.array(name_list))
    
    def add_bodies(self, masses, locations, velocities, names = None):
        if masses.shape[0] != locations.shape[0] or locations.shape[0] != velocities.shape[0]:
            raise ValueError("Size of arrays must be equal!")
        if names is None:
            names = np.full(masses.shape,"")

        if self._locations is None:
            self._locations = locations
        else:
            self._locations = np.concatenate([self._locations, locations])

        if self._velocities is None:
            self._velocities = velocities
        else:
            self._velocities = np.concatenate([self._velocities, velocities])

        self._masses = np.concatenate([self._masses,masses])
        self._names = np.concatenate([self._names,names])


    def run_simulation(self, time):
        self.history = np.array([])

        diff_eq = body_differentials(self._masses)
        integ = integrator.bulirsch_stoer(max_substeps = 100, diff_eq = diff_eq, h = 0.03, steps = 200)
        results, times = integ.integrate(state = self.get_state(), save_steps = True, initial_time = 0) 

        pass

    def animate(self, rate = 1):
        pass
    

if __name__ == "__main__":
    file_name = 'planets.csv'
    solar_system = system.from_file(file_name)
    solar_system.run_simulation(10)
    solar_system.animate()
