import numpy as np
import integrator
import gravity
import csv

class system(object):
    def __init__(self,animate = True):
        self.animate = animate
        self.gravity = gravity.particle_particle
        self.integrate = integrator.bulirsch_stoer
        self._locations = np.array([])
        self._masses = np.array([])
        self._velocities = np.array([])
        self._names = []
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

    def add_bodies_from_file(self, file_name):
        with open(file_name,'r') as csv_file:
            reader = csv.reader(csv_file)
            headers = reader.next()
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
                masses = np.array(mass_list),
                locations = np.array(loc_list),
                velocities = np.array(vel_list),
                names = np.array(name_list))
    
    def add_bodies(self, masses, locations, velocities, names = None):
        if masses.shape[0] != locations.shape[0] or locations.shape[0] != velocities.shape[0]:
            raise ValueError("Size of arrays must be equal!")
        if names is None:
            names = np.full(masses.shape,"")
        self._masses = np.concatenate([self._masses,masses])
        self._locations = np.concatenate([self._locations,locations])
        self._velocities = np.concatenate([self._velocities,velocities])
        self._names = names

    def run_simulation(self, time):
        self.history = np.array([])
        pass

    def animate(self, rate = 1):
        pass
    

if __name__ == "__main__":
    file_name = 'planets.csv'
    solar_system = system.from_file(file_name)
    system.run_simulation(10)
    system.animate()
