import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np


class FigureFrame(ttk.Frame):
    r"""
    Tk frame encapsulating a matplotlib figure and a axe.

    Abstraction for creating different types of plots.

    Parameters
    -----------
    plot_type: str, optional
        Available plots: Generic, Tafel, Complex Impedance, Complex Photocurrent. Default to Generic.
    kwargs: dict, optional
        keyword options for the tk frame.
    """
    def __init__(self, master, plot_type='Generic', **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.pack()

        s = ttk.Style()
        # s.theme_use('aqua')

        # variables
        if plot_type in ['Generic', 'Tafel', 'Complex Impedance', 'Complex Photocurrent']:
            self._plot_type = plot_type
        else:
            self._plot_type = 'Generic'

        self._fig = Figure()
        self._ax = self._fig.add_subplot(111)

        x = np.linspace(0, 5, 10)
        y = np.exp(x)
        self._lines = [Line2D(x, y, color='k', ls='', marker='.', ms=6, picker=100)]
        self._lines[0].set_visible(True)
        self._ax.add_line(self._lines[0])

        self._number_intervals = 0
        self._active_interval_index = 0
        self._intervals = list()
        self._artist_intervals = list()

        # frame widgets and grid properties
        nrows = 1
        for i in range(nrows):
            self.rowconfigure(index=i, weight=1)
        cols = 1
        for i in range(cols):
            self.columnconfigure(index=i, weight=1)

        self._top_frame = ttk.Frame(self)
        self._top_frame.grid(row=0, column=0, sticky='nswe')

        self._bottom_frame = ttk.Frame(self)
        self._bottom_frame.grid(row=1, column=0, sticky='nswe')
        nrows = 1
        for i in range(nrows):
            self._bottom_frame.rowconfigure(index=i, weight=1)
        ncols = 9
        for i in range(ncols):
            self._bottom_frame.columnconfigure(index=i, weight=1)

        # top frame => figure and toolbar
        container = self._top_frame
        self._canvas = FigureCanvasTkAgg(self._fig, master=container)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(side=tk.TOP, expand=tk.TRUE, fill=tk.BOTH)
        self._canvas.mpl_connect('pick_event', self._on_pick)
        self._canvas.mpl_connect('button_press_event', self._on_click)

        self._toolbar = NavigationToolbar2Tk(self._canvas, container)
        self._toolbar.update()
        self._canvas.get_tk_widget().pack(side=tk.TOP, expand=tk.TRUE, fill=tk.BOTH)

        # bottom frame => buttons for adding intervals
        row = 0
        col = 0
        container = self._bottom_frame

        ttk.Label(container, text='XY Scales:').grid(row=row, column=col, sticky='nswe')

        col += 1
        self._auto_aspect_bt = ttk.Button(container, text='Auto',
                                          command=self._aspect_auto)
        self._auto_aspect_bt.grid(row=row, column=col, sticky='nswe')

        col += 1
        self._auto_aspect_bt = ttk.Button(container, text='Equal',
                                          command=self._aspect_equal)
        self._auto_aspect_bt.grid(row=row, column=col, sticky='nswe')

        col += 1
        self._ylog_tkint = tk.IntVar()
        self._ylog_bt = ttk.Checkbutton(container, text='Y Log', command=self._ylog,
                                        variable=self._ylog_tkint)
        self._ylog_bt.grid(row=row, column=col, sticky='nswe')

        col += 1
        self._xlog_tkint = tk.IntVar()
        self._xlog_bt = ttk.Checkbutton(container, text='X Log', command=self._xlog,
                                        variable=self._xlog_tkint)
        self._xlog_bt.grid(row=row, column=col, sticky='nswe')

        row += 1
        col = 0
        ttk.Label(container, text='Intervals:').grid(row=row, column=col, sticky='nswe')

        col += 1
        self._add_bt = ttk.Button(container,
                                  text='Add',
                                  command=self._add_interval)
        self._add_bt.grid(row=row, column=col, sticky='nswe')

        col += 1
        self._add_bt = ttk.Button(container,
                                  text='Remove',
                                  command=self._remove_interval)
        self._add_bt.grid(row=row, column=col, sticky='nswe')

        col += 1
        self._next_bt = ttk.Button(container,
                                   text='Next',
                                   command=self._next_interval)
        self._next_bt.grid(row=row, column=col, sticky='nswe')

        col += 1
        self._back_bt = ttk.Button(container,
                                   text='Back',
                                   command=self._previous_interval)
        self._back_bt.grid(row=row, column=col, sticky='nswe')

        col += 1
        s.configure('Horizontal.TScale')
        self.scale_min = ttk.Scale(container, from_=0, orient=tk.HORIZONTAL, to=2.0,
                                   command=self._change_xmin)
        self.sliderlength = s.lookup('Horizontal.TScale', 'sliderlength')
        from_ = s.configure('Horizontal.TScale')
        print(from_)
        self.scale_min.grid(row=row, column=col, sticky='nswe')

        col += 1
        self.scale_max = ttk.Scale(container, from_=0.0, orient=tk.HORIZONTAL,
                                   to=2.0,
                                   command=self._change_xmax)
        self.scale_max.grid(row=row, column=col, sticky='nswe')

        row += 1
        s.configure('entry.TEntry')
        self._entry = ttk.Entry(container, style='entry.TEntry')
        self._entry.grid(row=row, column=col, sticky='nswe')

        # tweaks according to plot type
        if self._plot_type == 'Complex Impedance':
            self._aspect_equal()

        self.refresh()

    def _on_pick(self, event):
        r"""Method executed when a pick event occurs on a artist."""
        print('On Pick')
        for line in self._ax.get_lines():
            if 'Selected' in line.get_label():
                print('ii')
                line.set_markersize(line.get_markersize() / 1.5)
                print(line.get_label()[0:-11])
                line.set_label(line.get_label()[0:-11])
                break

        thisline = event.artist
        if isinstance(thisline, Line2D):
            print("Do something with the line.")
            # thisline.set_markersize(thisline.get_markersize()*1.5)
            # thisline.set_label(thisline.get_label() + ' - Selected')

        self.refresh()

    def _on_click(self, event):

        # right click left=1, middle=2, right=3, up (scroll event), down (scroll event)
        if event.button == 3:
            print(event.x, event.y, event.xdata, event.ydata)

    def _aspect_auto(self):
        self._ax.set_aspect('auto')
        self.refresh()

    def _aspect_equal(self):
        self._ax.set_aspect('equal')
        self.refresh()

    def _ylinear_scale(self):
        self._ax.set_yscale('linear')
        self.refresh()

    def _xlinear_scale(self):
        self._ax.set_xscale('linear')
        self.refresh()

    def _xlog(self):
        if self._xlog_tkint.get():
            self._ax.set_xscale('log')
        else:
            self._xlinear_scale()
        self.refresh()

    def _ylog(self):
        if self._ylog_tkint.get():
            self._ax.set_yscale('log')
        else:
            self._ylinear_scale()
        self.refresh()

    def _add_interval(self):
        self._number_intervals += 1
        self._active_interval_index = self._number_intervals - 1
        self._artist_intervals.append(self._ax.axvspan(0, 1.0, color='C7', alpha=0.1))
        self._update_min_max_scales()
        self._highlight_active_interval()
        self.refresh()

    def _remove_interval(self):
        if self._number_intervals:
            self._number_intervals -= 1
            if self._active_interval_index >= (self._number_intervals - 1):
                self._active_interval_index = self._number_intervals - 1

                self._artist_intervals[-1].remove()
                self._artist_intervals.pop(-1)
                self._update_min_max_scales()
                self._highlight_active_interval()
                self.refresh()

    def _next_interval(self):
        if self._number_intervals:
            if self._active_interval_index < (self._number_intervals - 1):
                self._active_interval_index += 1
            else:
                self._active_interval_index = 0
            self._update_min_max_scales()
            self._highlight_active_interval()
            self.refresh()

    def _previous_interval(self):
        if self._number_intervals:
            if self._active_interval_index > (-self._number_intervals + 1):
                self._active_interval_index -= 1
            else:
                self._active_interval_index = 0

            xmin, xmax = self._current_interval_limits()
            self.scale_min.set(xmin)
            self.scale_max.set(xmax)
            self._highlight_active_interval()
            self.refresh()

    def _highlight_active_interval(self):

        for i in range(self._number_intervals):
            self._artist_intervals[i].set_alpha(0.05)

        if self._number_intervals:
            self._artist_intervals[self._active_interval_index].set_alpha(0.1)

    def _change_xmin(self, value):
        if self._number_intervals:
            xy = self._artist_intervals[self._active_interval_index].get_xy()
            xy[0, 0] = value
            xy[1, 0] = value
            xy[4, 0] = value
            self.refresh()

    def _change_xmax(self, value):
        if self._number_intervals:
            xy = self._artist_intervals[self._active_interval_index].get_xy()
            xy[2, 0] = value
            xy[3, 0] = value
            self.refresh()

    def _update_min_max_scales(self):

        xmin, xmax = self._current_interval_limits()
        self.scale_min.set(xmin)
        self.scale_max.set(xmax)

    def _current_interval_limits(self):
        xy = self._artist_intervals[self._active_interval_index].get_xy()
        return xy[0, 0], xy[2, 0]

    @property
    def canvas(self):
        return self._canvas

    @property
    def toolbar(self):
        return self._toolbar

    @property
    def figure(self):
        return self._fig

    @property
    def number_intervals(self):
        return self._number_intervals

    @property
    def active_interval_index(self):
        return self._active_interval_index

    @property
    def active_interval(self):
        r"""Return the active interval which is a matplotlib polygon."""
        return self._artist_intervals[self._active_interval_index]

    def refresh(self):

        self._ax.relim()
        self._ax.autoscale()
        self._ax.autoscale_view()

        self._highlight_active_interval()

        self._canvas.draw()


class App(tk.Frame):
    r"""
    Main app window from which the FigureFrame widget will be used.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.master = master
        self.root = master
        self.pack(side=tk.TOP, expand=tk.TRUE, fill=tk.BOTH)

        for i in range(0, 1):
            self.rowconfigure(index=i, weight=1)

        self.columnconfigure(index=0, weight=1)

        container = self
        self.figure = FigureFrame(container)
        self.figure.grid(row=0, column=0, sticky='nswe')

    def start(self):
        self.master.mainloop()

    def quit(self):
        self.master.quit()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    app.start()
