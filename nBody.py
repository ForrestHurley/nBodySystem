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
        dv = self.gravity.get_acceleration(bodies=loc)
        dx = vel
        return system.loc_vel_to_state(dx,dv)

class ephemerides_rails(integrator.differential_equation):
    def __init__(self, ephemerides, *args, **kwargs):
        super().__init__()
        self.ephemerides = ephemerides
        self.gravity = gravity.particle_rail(masses = ephemerides.masses, *args, **kwargs)

    def evaluate(self, state, constant_args = (), time = 0):
        eph_locs = self.ephemerides.positions(time)
        loc, vel = system.state_to_loc_vel(state)
        dv = self.gravity.get_acceleration(particles=loc,bodies=eph_locs)
        dx = vel
        return system.loc_vel_to_state(dx,dv)

class system(object):
    def __init__(self):
        self.gravity = gravity.particle_particle
        self._locations = None
        self._masses = np.array([])
        self._velocities = None
        self._names = np.array([])
        self._history = []
        self.h = 0.1
        self.default_integ = integrator.rk4()

    @classmethod
    def from_mass_state(cls, masses, locations, velocities, *args, **kwargs):
        new_system = cls(*args, **kwargs)
        new_system.add_bodies(masses, locations, velocities)
        return new_system

    @classmethod
    def from_file(cls, file_name, verbose = False, *args, **kwargs):
        new_system = cls(*args, **kwargs)
        new_system.add_bodies_from_file(file_name)
        if verbose:
            print(new_system)
        return new_system

    @classmethod
    def from_ephemerides(cls, file_name, objects = None, start_time = None, verbose = False, *args, **kwargs):
        ephem_data = ephemerides(file_name)
        if not objects is None:
            ephem_data.limit_objects(objects)

        if start_time is None:
            start_time = time.time()
        state_array = ephem_data.state(start_time)
        mass_array = ephem_data.masses
        name_array = ephem_data.names
    
        new_system = cls.from_state(state_array, mass_array, name_array, *args, **kwargs)
        if verbose:
            print(new_system)

        return new_system

    @classmethod
    def from_state(cls, state, masses, names = None, *args, **kwargs):
        new_system = cls(*args, **kwargs)
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

    @property
    def diff_eq(self):
        return differential_body(self._masses,
            max_acceleration = 1e20)

    def run_simulation(self, total_time, integ = None, verbose = False):

        diff_eq = self.diff_eq

        if integ is None:
            integ = self.default_integ
        integ.diff_eq = diff_eq
        integ.h = self.h
        integ.steps = int(total_time / self.h)
        integ.verbose = verbose


        results, times = integ.integrate(state = self.get_state(),
            save_steps = True, initial_time = 0) 

        self._history = [times, np.array(results)]
        return self._history

    def draw(self, rate = -1):
        positions, velocities = system.state_to_loc_vel(self._history[1])

        positions = np.transpose(positions, (1, 0, 2))
        self._call_plotter(positions, rate = rate)

    def _call_plotter(self, positions, rate = -1):
        if rate < 0:
            try:
                self.plot3d.plot(positions, show = True)
            except AttributeError:
                self.plot3d = plotter()
                self.plot3d.plot(positions, show = True)
        else:
            try:
                self.plot3d_anim(positions, show = True)
            except AttributeError:
                self.plot3d_anim = anim_plotter()
                self.plot3d_anim.plot(positions, show = True)

class railed_system(system):
    def __init__(self,ephemerides,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._ephemerides = ephemerides
        self.default_integ = integrator.adams_moulton4()

    @property
    def diff_eq(self):
        return ephemerides_rails(ephemerides = self._ephemerides,
            max_acceleration = 1e20)

    def draw(self, rate = -1):
        positions, velocities = system.state_to_loc_vel(self._history[1])
        times = self._history[0]
        
        positions = np.transpose(positions, (1, 0, 2))
        rail_paths = self._ephemerides.path(times = times)
        
        draw_data = np.concatenate((positions, rail_paths), axis = 0)
        self._call_plotter(positions = draw_data, rate = rate)

def rocket_system(railed_system):
    def __init__(self, ephemerides, start_time = None, rocket_count = 1, rocket_states = None, *args, **kwargs):
        if not rocket_states is None:
            rocket_count = rocket_states.shape[0]
        else:
            rocket_states = np.array([ephemerides.object_state(399,start_time)]*rocket_count)
        super().__init__(masses = np.array([0]*rocket_count), *args, **kwargs)

    def run_simulation(self, *args, **kwargs):
        pass

if __name__ == "__main__":
    file_name = '../Data/solarSystem.txt'
    eph_data = ephemerides.LimitObjects(file_name,range(10,2000))
    solar_system = railed_system.from_mass_state(
            masses = np.array([0]*1000),
            locations = np.array([[1e8,1e8,1e8]]*1000),
            velocities = np.array([[20,10,5]]*1000),
            ephemerides = eph_data)
    solar_system.h = 60*60*6
    result = solar_system.run_simulation(60*60*24*365, verbose = True)
    solar_system.draw(rate = -1)
