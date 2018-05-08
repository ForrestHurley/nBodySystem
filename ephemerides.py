import spiceypy
import itertools

class ephemerides(object):
    def __init__(self,file_name):
        spice.furnsh(file_name)

    @property
    def spk_list(self):
        try:
            return self._spk_list
        except AttributeError:

            total_count = spice.ktotal("spk")
            self._spk_list = [spice.kdata(i, "spk")[0] for i in range(total_count)]

            return self._spk_list

    @property
    def object_list(self):
        try:
            return self._object_list
        except AttributeError:
            
            object_generator = (spice.spkobj(filename) for filename in self.spk_list)
            self._object_list = list(set(itertools.chain.from_iterable(file_objects)))

            return self._object_list

    @property
    def masses(self):
        try:
            return self._masses
        except AttributeError:
                self._masses = \
                    [spice.bodvrd(str(obj),"GM",1)[1] for obj in self.object_list if spice.bodfnd(obj,"GM")] 
            return self._masses

    @property
    def names(self):
        try:
            return self._names
        except AttributeError:
            self._names = [spice.bodc2n(obj,33) for obj in self.object_list]
            return self._names

    def positions(self, time):
        for body in self.object_list:
            
