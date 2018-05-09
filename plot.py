import numpy as np
from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation

class plotter(object):
    def __init__(self):
        self.x_range = (-2, 2)
        self.y_range = (-2, 2)
        self.z_range = (-2, 2)
        
        self.labels = ['X', 'Y', 'Z']
        self.title = "Orbits"

    def plot(self, data, show = True, set_axes = False): #data is bodies, times, position (x y z)
        self.data = data
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        self.ln_plots = []
        
        for datum in self.data:
            self._draw_plot(ax, datum)

        if set_axes:
            ax.set_xlim3d(self.x_range)
            ax.set_ylim3d(self.y_range)
            ax.set_zlim3d(self.z_range)

        ax.set_xlabel(self.labels[0])
        ax.set_ylabel(self.labels[1])
        ax.set_zlabel(self.labels[2])

        ax.set_title(self.title)

        if show:
            plt.show()

        return fig

    def _draw_plot(self, ax, datum):
        self.ln_plots.append(ax.plot(datum[:,0],datum[:,1],datum[:,2]))

class anim_plotter(plotter):
    def __init__(self, history = 100, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = history

    def update(self,n):
        start = max(n - self.history, 0)
        for body, datum in zip(self.pt_plots, self.data):
            body.set_data(datum[n][:2])
            body.set_3d_properties(datum[n][2])
        for body, datum in zip(self.ln_plots, self.data):
            body.set_data(datum[start:n,0],datum[start:n,1])
            body.set_3d_properties(datum[start:n,2])
        return self.ln_plots

    def plot(self, data, *args, **kwargs):
        kwargs['show'] = False
        
        self.pt_plots = []
        fig = super().plot(data, *args, **kwargs)

        count = self.data.shape[1]

        print(count)

        ani = animation.FuncAnimation(fig, self.update, count, interval = 50, blit = False)

        plt.show()
        return fig

    def _draw_plot(self, ax, datum):
        self.pt_plots.append(ax.plot([datum[0,0]],[datum[0,1]],[datum[0,2]],marker = 'o')[0])
        self.ln_plots.append(ax.plot([datum[0,0]],[datum[0,1]],[datum[0,2]])[0])
