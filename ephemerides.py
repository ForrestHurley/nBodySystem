import spiceypy as spice
import numpy as np
import itertools
import time

class ephemerides(object):
    def __init__(self,file_name):
        spice.furnsh(file_name)
        self._nullify_derived()

    @classmethod
    def LimitObjects(cls,file_name,objects):
        cls_object = cls(file_name)
        cls_object.limit_objects(objects)
        return cls_object

    @property
    def spk_list(self):
        try:
            return self._spk_list
        except AttributeError:

            total_count = spice.ktotal("spk")
            self._spk_list = [spice.kdata(i, "spk")[0] for i in range(total_count)]

            return self._spk_list

    def _nullify_derived(self):
        self._masses = None
        self._names = None
        self._radii = None

    def limit_objects(self,objects):
        self._nullify_derived()
        self._object_list = np.intersect1d(self.object_list,objects)

    @property
    def object_list(self):
        try:
            return self._object_list
        except AttributeError:
            
            object_generator = (spice.spkobj(filename) for filename in self.spk_list)
            self._object_list = np.array(list(set(itertools.chain.from_iterable(object_generator))))

            return self._object_list

    @property
    def masses(self):
        if not self._masses is None:
            return self._masses
        else:
            self._masses = \
                np.array(
                [spice.bodvrd(str(obj),"GM",1)[1] if spice.bodfnd(obj, "GM") else 0 for obj in self.object_list]
                ) 
            return self._masses

    @property
    def radii(self):
        if not self._radii is None:
            return self._radii
        else:
            self._radii = \
                np.array(
                [spice.bodvrd(str(obj),"RADII",3)[1] if spice.bodfnd(obj,"RADII") else 0 for obj in self.object_list]
                )
            return self._radii

    @property
    def names(self):
        if not self._names is None:
            return self._names
        else:
            self._names = [spice.bodc2n(obj,33) for obj in self.object_list]
            return self._names

    def positions(self, time):
        pos_array = np.array([spice.spkpos(str(obj), time, 'J2000', 'NONE', '0')[0] for obj in self.object_list])
        return np.array(pos_array)

    def state(self, time):
        state_array = ([spice.spkezr(str(obj), time, 'J2000', 'NONE', '0')[0] for obj in self.object_list]) 
        return np.reshape(state_array,(-1,2,3))

    def velocities(self, time):
        state = self.state(time)
        return state[:,1]

    def path(self, times = None, years = 10, steps = 1000):
        delta_time = years * 60 * 60 * 24 * 365.25
        if times is None:
            current_time = time.time()
            times = np.linspace(
                start = current_time,
                stop = current_time + delta_time,
                num = steps)
        position_array = np.array([self.positions(timestamp) for timestamp in times])
        path = np.transpose(position_array,(1,0,2))
        return path

if __name__ == "__main__":
    from plot import plotter, anim_plotter
    body_data = ephemerides.LimitObjects("../Data/solarSystem.txt",[10,199,299,399,499,599,699,799,899,999])
    paths = body_data.path()
    print(paths.shape)
    plot3d = anim_plotter()
    plot3d.plot(paths, show = True)
