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

    def plot(self, data, show = True): #data is bodies, times, position (x y z)
        self.data = data
        fig = plt.figure()
        ax = p3.Axes3D(fig)

        self.ax_plots = []
        
        for datum in self.data:
            self._draw_plot(ax, datum)

        ax.set_xlim3d(self.x_range)
        ax.set_xlabel(self.labels[0])
        ax.set_ylim3d(self.y_range)
        ax.set_ylabel(self.labels[1])
        ax.set_zlim3d(self.z_range)
        ax.set_zlabel(self.labels[2])

        ax.set_title(self.title)

        if show:
            plt.show()

        return fig

    def _draw_plot(self, ax, datum):
        self.ax_plots.append(ax.plot(datum[:,0],datum[:,1],datum[:,2]))

class anim_plotter(plotter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self,n):
        for body, datum in zip(self.ax_plots, self.data):
            body.set_data(datum[n][:2])
            body.set_3d_properties(datum[n][2])
        return self.ax_plots

    def plot(self, data, *args, **kwargs):
        kwargs['show'] = False
        fig = super().plot(data, *args, **kwargs)

        count = self.data.shape[1]

        print(count)

        ani = animation.FuncAnimation(fig, self.update, count, interval = 50, blit = False)

        plt.show()
        return fig

    def _draw_plot(self, ax, datum):
        self.ax_plots.append(ax.plot([datum[0,0]],[datum[0,1]],[datum[0,2]],marker = 'o')[0])
