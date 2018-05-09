import numpy as np
import integrator
import gravity
from ephemerides import ephemerides
import csv
import time
from plot import plotter, anim_plotter

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
        self.h = 0.1

    @classmethod
    def from_mass_state(cls, masses, locations, velocities):
        new_system = cls()
        new_system.add_bodies(masses, locations, velocities)
        return new_system

    @classmethod
    def from_file(cls, file_name, verbose = False):
        new_system = cls()
        new_system.add_bodies_from_file(file_name)
        if verbose:
            print(new_system)
        return new_system

    @classmethod
    def from_ephemerides(cls, file_name, objects = None, start_time = None, verbose = False):
        ephem_data = ephemerides(file_name)
        if not objects is None:
            ephem_data.limit_objects(objects)

        if start_time is None:
            start_time = time.time()
        state_array = ephem_data.state(start_time)
        mass_array = ephem_data.masses
        name_array = ephem_data.names
    
        new_system = cls.from_state(state_array, mass_array, name_array)
        if verbose:
            print(new_system)

        return new_system

    @classmethod
    def from_state(cls, state, masses, names = None):
        new_system = cls()
        new_system.set_masses(masses)
        new_system.set_state(state)
        new_system.set_names(names)
        return new_system

    def __str__(self):
        out = "\n".join(
            [" ".join([str(name),str(mass),str(loc),str(vel)]) \
                for name, mass, loc, vel \
                in zip(self._names, self._masses, self._locations, self._velocities)])
        return out

    def set_state(self, state):
        self._locations, self._velocities = \
            system.state_to_loc_vel(state)

    def set_masses(self, masses):
        self._masses = masses

    def set_names(self, names):
        self._names = names

    def get_state(self):
        return system.loc_vel_to_state(self._locations, self._velocities)

    def get_loc_vel(self):
        return self._locations, self._velocities
    
    def get_masses(self):
        return self._masses

    @staticmethod
    def state_to_loc_vel(state):
        loc = np.take(state, 0, axis = -2)
        vel = np.take(state, 1, axis = -2)
        return loc, vel

    @staticmethod
    def loc_vel_to_state(loc, vel):
        return np.stack([loc, vel], axis = -2)

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


    def run_simulation(self, time, verbose = False):
        diff_eq = body_differentials(self._masses,
            max_acceleration = 1e30)
        integ = integrator.bulirsch_stoer(
            max_substeps = 100, 
            diff_eq = diff_eq, 
            h = self.h, 
            steps = int(time / self.h),
            error_tolerance = 1e-5,
            ignore_overruns = True,
            verbose = verbose)
        results, times = integ.integrate(state = self.get_state(), save_steps = True, initial_time = 0) 

        self._history = [times, np.array(results)]
        return self._history

    def draw(self, rate = -1):
        positions, velocities = system.state_to_loc_vel(self._history[1])

        positions = np.transpose(positions, (1, 0, 2))

        try:
            self.plot3d.plot(positions, show = True)
        except AttributeError:
            self.plot3d = plotter()
            self.plot3d.plot(positions, show = True)

if __name__ == "__main__":
    '''
    file_name = 'planets.csv'
    solar_system = system.from_file(file_name, verbose = True)
    '''
    file_name = '../Data/solarSystem.txt'
    solar_system = system.from_ephemerides(file_name,list(range(1,11)), verbose = False)
    
    solar_system.h = 60*60*12
    result = solar_system.run_simulation(60*60*24*28*2, verbose = True)
    solar_system.draw()
