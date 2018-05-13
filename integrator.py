import numpy as np
from lagrange import lagrange
import sys

class differential_equation(object):
    def __call__(self, state, diff_eq_args = (), time = 0):
        return self.evaluate(state, diff_eq_args, time = time)

    def evaluate(self, state, const_args = (), time = 0):
        return np.zeros(state.shape)

class integrator(object):
    def __init__(self, diff_eq = differential_equation(), h=1, steps=10, verbose = False):
        self.h = h
        self.steps = steps
        self.diff_eq = diff_eq
        self.verbose = verbose

    def integrate(self, state, save_steps = False, initial_time = 0, diff_eq_args = ()):
        if self.verbose:
            print("Starting integration")
        if save_steps:
            state_list = []
            time_list = []
        for i in range(self.steps):
            step_time = initial_time + i * self.h
            state = self.general_step(state = state, 
                        time = step_time, 
                        diff_eq_args = diff_eq_args)

            state_list.append(state)
            time_list.append(initial_time + i * self.h)
            if self.verbose:
                sys.stdout.write("\033[K")
                print("Finished iteration {0}".format(i),end="\r")
                sys.stdout.flush()

        if save_steps:
            return state_list, time_list
        return state

    def general_step(self,*args,**kwargs):
        return self.step(*args, **kwargs)

    def step(self, state, time = 0, diff_eq_args = ()):
        pass

class event_integrator(integrator):
    def __init__(self, discrete_events = [], *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.discrete_events = discrete_events

    def integrate(self, *args, **kwargs):
        self.discrete_index = 0
        return super().integrate(*args, **kwargs)

    def general_step(self,time = 0,*args,**kwargs):
        new_state = self.step(time = time, *args, **kwargs)

        if len(self.discrete_events) > self.discrete_index:
            print(time)
            if time > self.discrete_events[self.discrete_index][0]:
                new_state += self.discrete_events[self.discrete_index][1]
                self.discrete_index += 1
        
                print("Activated")
        return new_state

class euler(event_integrator):
    def step(self, state, time = 0, diff_eq_args = ()):
        rates = self.diff_eq(state, diff_eq_args, time = time)
        return state + rates * self.h

class trapezoidal(event_integrator):
    pass

class rk2(event_integrator):
    def step(self, state, time = 0, diff_eq_args = ()):
        k1 = self.h * self.diff_eq(state, *diff_eq_args, time = time)
        k2 = self.h * self.diff_eq(state + k1,
                                diff_eq_args,
                                time = time + self.h)
        return state + ( (k1 + k2) / 2 )

class rk3(event_integrator):
    def step(self, state, time = 0, diff_eq_args = ()):
        k1 = self.h * self.diff_eq(state, *diff_eq_args, time = time)
        k2 = self.h * self.diff_eq(state + 0.5 * k1,
                                diff_eq_args,
                                time = time + 0.5 * self.h)
        k3 = self.h * self.diff_eq(state - k1 + 2 * k2,
                                diff_eq_args,
                                time = time + self.h)
        return state + ( (k1 + 4 * k2 + k3) / 6 )

class rk4(event_integrator):
    def step(self, state, time = 0, diff_eq_args = ()):
        k1 = self.h * self.diff_eq(state, diff_eq_args, time = time)
        k2 = self.h * self.diff_eq(state + 0.5 * k1,
                                diff_eq_args,
                                time = time + 0.5 * self.h)
        k3 = self.h * self.diff_eq(state + 0.5 * k2,
                                diff_eq_args,
                                time = time + 0.5 * self.h)
        k4 = self.h * self.diff_eq(state + k3,
                                diff_eq_args,
                                time = time + self.h)
        return state + ( (k1 + 2 * k2 + 2 * k3 + k4) / 6 )

class adams_bashforth4(rk4):
    def integrate(self, state, initial_time = 0, diff_eq_args = (), *args, **kwargs):
        
        self.diff_list = [self.diff_eq(
            state = state, time = initial_time, diff_eq_args = diff_eq_args)]

        for i in range(3):
            state = super().step(
                    state = state,
                    time = initial_time + i * self.h,
                    diff_eq_args = diff_eq_args)

            self.diff_list.append(self.diff_eq(
                state = state,
                time = initial_time + i * self.h,
                diff_eq_args = diff_eq_args))

        state = super().integrate(state,
            initial_time = initial_time + 4 * self.h,
            diff_eq_args = diff_eq_args,
            *args, **kwargs)

        return state

    def step(self, state, time = 0, diff_eq_args = ()):
            del self.diff_list[0]
            self.diff_list.append(self.diff_eq(
                state = state, time = time, diff_eq_args = diff_eq_args))

            return state \
                + self.h / 24 * ( 55 * self.diff_list[-1] \
                - 59 * self.diff_list[-2] + 37 * self.diff_list[-3] \
                - 9 * self.diff_list[-4])

class adams_moulton4(adams_bashforth4):
    def step(self, state, time = 0, diff_eq_args = ()):
        next_state = super().step(
            state = state,
            time = time,
            diff_eq_args = diff_eq_args)

        new_diff = self.diff_eq(next_state, diff_eq_args, time = time + self.h)

        adjusted_state = state + self.h / 24. * (
            9. * new_diff + 19 * self.diff_list[-1] - 5 * self.diff_list[-2] + self.diff_list[-3] )

        return adjusted_state

class modified_midpoint(event_integrator):
    def __init__(self, max_substeps = 30, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if max_substeps < 1:
            raise ValueError("Substeps must be at least 1")
        self._max_substeps = max_substeps

    def step(self, state, time = 0, diff_eq_args = ()):
        return self.midpoint(state, 
                        substeps = self._max_substeps, 
                        time = time, 
                        diff_eq_args = diff_eq_args)

    def midpoint(self, state, substeps = 10, time = 0, diff_eq_args = ()):

        h = self.h / substeps

        two_previous = state
        last_state = state + h * self.diff_eq(state,
            diff_eq_args,
            time = time)

        for i in range(2, substeps + 1):
            new_state = two_previous \
                + 2 * h * self.diff_eq(last_state,
                    diff_eq_args,
                    time = time + (i - 1) * h)

            two_previous = last_state
            last_state = new_state

        final_state = (last_state + two_previous +
            h * self.diff_eq(last_state,
                diff_eq_args,
                time = time + self.h)) / 2
        
        return final_state

def is_close(a, b, abs_tol = 1e-12):
    return (a - b < abs_tol) & (a - b > -abs_tol)

class bulirsch_stoer(modified_midpoint):
    def __init__(self,initial_substeps = 10, ignore_overruns=False,error_tolerance=1e-12,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.ignore_overruns = ignore_overruns
        self.error_tolerance = error_tolerance
        self.initial_substeps = initial_substeps
    def step(self, state, time = 0, diff_eq_args = ()):
        
        polynomial = lagrange(eval_x = 0)
        last_x = float('Inf')

        for i in range(int(self.initial_substeps / 2), int(self._max_substeps / 2)):
            estim = self.midpoint(state,
                substeps = 2 * i,
                time = time,
                diff_eq_args = diff_eq_args)

            polynomial.add_point(x = self.h / ( 2 * i ), y = estim) 

            new_x = polynomial.estimate
            if np.all(is_close(new_x, last_x, abs_tol = self.error_tolerance)):
                return new_x
            else:
                last_x = new_x

        if not self.ignore_overruns:
            raise ValueError("Exceeded max iterations")
        return last_x

class test_equation(differential_equation):
    k = np.array([-1,2])
    def evaluate(self, state, constant_args = (), time = 0):
        return np.flip(state,axis=0)*test_equation.k + time

if __name__ == "__main__":
    equation = test_equation()

    integ = adams_moulton4(diff_eq = equation, h = 0.3, steps = 10)
    #integ = modified_midpoint(max_substeps = 5, diff_eq = equation, h = 0.03, steps = 100)
    #integ = bulirsch_stoer(max_substeps = 100, diff_eq = equation, ignore_overruns = True, h = 0.03, steps = 100)
    results, times = integ.integrate(state = np.array([1, 3]),save_steps = True,initial_time = 0)
    from matplotlib import pyplot as plt
    plt.plot(times, np.array(results))
    plt.show()
