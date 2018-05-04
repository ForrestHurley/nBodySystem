
import importlib
if importlib.util.find_spec("numpy") is None:
    from math import isclose
else:
    from numpy import isclose

class differential_equation(object):
    def __call__(self, state, *const_args, time = 0):
        self.evaluate(state, *const_args, time = time)

    def evaluate(self, state, *const_args, time = 0):
        pass

class integrator(object):
    def __init__(self, diff_eq, h=1, steps=10):
        self.h = h
        self.steps = steps
        self.diff_eq = diff_eq

    def integrate(self, state, save_steps = False, initial_time = 0, diff_eq_args = ()):
        if save_steps:
            state_list = []
        for i in range(self.steps):
            state = step(self, 
                        state = state, 
                        time = initial_time + i * h, 
                        diff_eq_args = diff_eq_args)
            state_list.append(state)

        if save_steps:
            return state_list
        return state

    def step(self, state, time = 0, diff_eq_args = ()):
        pass

class euler(integrator):
    def step(self, state, time = 0, diff_eq_args = ()):
        rates = self.diff_eq(state, *diff_eq_args, time = time)
        return state + rates * self.h

class trapezoidal(integrator):
    pass

class rk2(integrator):
    def step(self):
        k1 = self.h * self.diff_eq(state, *diff_eq_args, time = time)
        k2 = self.h * self.diff_eq(state + k1,
                                *diff_eq_args,
                                time = time + self.h)
        return state + ( (k1 + k2) / 2 )

class rk3(integrator):
    def step(self, state, time = 0, diff_eq_args = ()):
        k1 = self.h * self.diff_eq(state, *diff_eq_args, time = time)
        k2 = self.h * self.diff_eq(state + 0.5 * k1,
                                *diff_eq_args,
                                time = time + 0.5 * self.h)
        k3 = self.h * self.diff_eq(state - k1 + 2 * k2,
                                *diff_eq_args,
                                time = time + self.h)
        return state + ( (k1 + 4 * k2 + k3) / 6 )

def rk4(integrator):
    def step(self, state, time = 0, diff_eq_args = ()):
        k1 = self.h * self.diff_eq(state, *diff_eq_args, time = time)
        k2 = self.h * self.diff_eq(state + 0.5 * k1,
                                *diff_eq_args,
                                time = time + 0.5 * self.h)
        k3 = self.h * self.diff_eq(state + 0.5 * k2,
                                *diff_eq_args,
                                time = time + 0.5 * self.h)
        k3 = self.h * self.diff_eq(state + k3,
                                *diff_eq_args,
                                time = time + self.h)
        return state + ( (k1 + 2 * k2 + 2 * k3 + k4) / 6 )

class modified_midpoint(integrator):
    def __init__(self, max_substeps = 30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if max_substeps < 1:
            raise ValueError("Substeps must be at least 1")
        self._max_substeps = max_substeps

    def step(self, state, time = 0, diff_eq_args = ()):
        return midpoint(self, state, 
                        substeps = self._max_substeps, 
                        time = time, 
                        diff_eq_args = diff_eq_args)

    def substep_midpoint(self, last_state, two_previous_state, time, diff_eq_args = ()):
        return two_previous + 
            2 * h * self.diff_eq(last_state,
                *diff_eq_args,
                time = time)

    def midpoint(self, state, substeps = 10, time = 0, diff_eq_args = ()):

        h = self.h / substeps

        two_previous = state
        last_state = state + h * self.diff_eq(state,
            *diff_eq_args,
            time = time)

        for i in range(2, substeps + 1):
            new_state = self.substep_midpoint(last_state,
                two_previous,
                time = time + (i - 1) * h,
                diff_eq_args = diff_eq_args)

            two_previous = last_state
            last_state = new_state

        final_state = (last_state + two_previous +
            h * self.diff_eq(last_state,
                *diff_eq_args,
                time = time + self.h)) / 2
        
        return final_state

class bulirsch_stoer(modified_midpoint):
    def step(self, state, time = 0, diff_eq_args = ()):
        
        polynomial = point_lagrange(eval_x = 0)
        last_x = float('Inf')

        for i in range(self._max_substeps):
            estim = self.midpoint(state,
                substeps = 2 * n,
                time,
                diff_eq_args = diff_eq_args)

            polynomial.add_point(x = self.h / ( 2 * n), y = estim) 

            new_x = polynomial.at_x
            if isclose(new_x, last_x, abs_tol = 1e-12):
                return new_x
            else:
                last_x = new_x

        raise ValueError("Exceeded max iterations")
