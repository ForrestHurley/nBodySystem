class lagrange(object):
    def __init__(self, eval_x = 0):
        self._eval_x = eval_x
        self._extrapolations = []

    def add_point(self, x, y):
        new_extraps = [(y, x)]
        for past_extrap, x_old in self._extrapolations:
            new_val = ((self._eval_x - x) * past_extrap \
                + (x_old - self._eval_x) * new_extraps[-1][0])\
                / (x_old - x)
            new_extraps.append((new_val, x_old))
        self._extrapolations = new_extraps
        return self.estimate

    @property
    def estimate(self):
        return self._extrapolations[-1][0]

if __name__ == "__main__":
    interpolator = lagrange(eval_x = 0)
    print(interpolator.add_point(1,2))
    print(interpolator.add_point(0.5,3))
    print(interpolator.add_point(0.25,3.75))
    print(interpolator.add_point(0.125,4.25))
    print(interpolator.add_point(0.0625,4.5))
