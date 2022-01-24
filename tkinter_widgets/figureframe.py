r"""
Copyright (C) 2020-2021 Milan Skocic.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author: Milan Skocic <milan.skocic@gmail.com>
"""
from typing import Tuple
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from numpy.typing import ArrayLike

class PlotFrame(ttk.Frame):
    r"""
    Class for encapsulating a matplotlib figure and a toolbar. See __init__.__doc__"""
    def __init__(self, master, x: Tuple[ArrayLike], y: Tuple[ArrayLike], labels:Tuple[str], **kwargs):
        r"""
        Tk frame encapsulating a matplotlib figure and a toolbar.

        Parameters
        ------------
        x: tuple
            Tuple of x data.
        y: tuple
            Tuple of y data.
        labels: tuple
            Tuple of data labels.
        kwargs: dict, optional
            Keyword arguments for the tk frame.
        """
        super().__init__(master, **kwargs)
        self.master = master

        self.figure = Figure()
        self.subplot = self.figure.add_subplot(111)

        # frame widgets and grid properties
        nrows = 1
        for i in range(nrows):
            self.rowconfigure(index=i, weight=1)
        cols = 1
        for i in range(cols):
            self.columnconfigure(index=i, weight=1)

        nrows = 2
        for i in range(nrows):
            self.rowconfigure(index=i, weight=1)
        ncols = 0
        for i in range(ncols):
            self.columnconfigure(index=i, weight=1)

        # top frame => figure and toolbar
        container = self
        self.canvas = FigureCanvasTkAgg(self.figure, master=container)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=tk.TRUE, fill=tk.BOTH)

        self.toolbar = NavigationToolbar2Tk(self.canvas, container)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, expand=tk.TRUE, fill=tk.BOTH)

        for xdata, ydata, label in zip(x, y, labels):
            self.subplot.plot(xdata, ydata, marker=".", label=label)

        self.subplot.legend()

    def refresh(self):
        r"""
        Refresh the plot.
        """
        self.subplot.relim()
        self.subplot.autoscale()
        self.subplot.autoscale_view()
        self.canvas.draw()