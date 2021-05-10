r"""
Simple multiprocess example for plot and fitting data with tkinter.
"""
import queue as Queue
from tkinter import *
from tkinter.ttk import *
import multiprocessing as mp

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from scipy import optimize


class GUI(Frame):
    r"""Class documentation should be in __init__ method. PEP257."""
    def __init__(self, master):
        r"""Main tkinter app.

        Parameters
        ------------
        kwargs: dict, optional
            Tkinter frame kwargs.
        """
        super().__init__(master)
        self.master = master
        self.pack()
        self.label = Label(self, text='Process label')
        self.label.pack()
        self.start_bt = Button(self, text='Run Processes', command=self.on_start_bt)
        self.start_bt.pack()
        self.stop_bt = Button(self, text='Stop Processes', command=self.on_stop_bt)
        self.stop_bt.configure(state=DISABLED)
        self.stop_bt.pack()
        self.process_list = []
        self.f = Figure(figsize=(5, 4), dpi=100)
        self.a = self.f.add_subplot(211)
        self.b = self.f.add_subplot(212)
        self.canvas = FigureCanvasTkAgg(self.f, self)

        self.canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas._tkcanvas.pack(side=BOTTOM, fill=BOTH, expand=1)

        self.num_process = 2
        self.num_fit = 2
        self.num_run = 20
        for i in range(self.num_fit - 1):
            self.a.axvline(i + self.num_run - 0.5, color='k', linewidth=2)
        self.nbfits_x = []
        self.nbfits_errors = []
        self.running_proc = [0] * self.num_fit

        self.x_exp = np.linspace(0, np.pi / 2, 100)
        self.y_exp = self.model([1, 1], self.x_exp) + np.random.normal(0, 0.01, self.x_exp.size)
        self.p0 = [1.0, 1.0]

        self.display = []
        for i in range(self.num_fit):
            self.display.append(Label(self, text='Parameters: ' + str(self.p0)))

        for i in self.display:
            i.pack(side=BOTTOM)

        self.queue = mp.Queue()

        self.line_exp = Line2D(self.x_exp, self.y_exp, color='k', linestyle='none', marker='o', mfc='w',
                               fillstyle='none', label='Exp')
        self.line_fit = Line2D([], [], color='r', linestyle='-', marker='None', mfc='w', fillstyle='none', label='Fit1',
                               alpha=0.5)
        self.line_fit2 = Line2D([], [], color='g', linestyle='-', marker='None', mfc='w', fillstyle='none',
                                label='Fit2')
        self.line_error = Line2D([], [], color='m', linestyle='none', marker='o', mfc='w', fillstyle='none',
                                 label='Error')
        self.b.add_line(self.line_exp)
        self.b.add_line(self.line_fit)
        self.b.add_line(self.line_fit2)
        self.a.add_line(self.line_error)
        self.a.set_yscale('log')
        self.refresh()

    def refresh(self):
        self.a.set_autoscale_on(True)
        self.a.relim()
        self.a.autoscale_view(True, True, True)
        self.a.legend(loc='best')

        self.b.set_autoscale_on(True)
        self.b.relim()
        self.b.autoscale_view(True, True, True)
        self.b.legend(loc='best')
        self.canvas.draw()

    def model(self, p, x_exp):
        y_calc = p[0] * np.sin(2 * np.pi * x_exp * p[1])
        return y_calc

    def residuals(self, p, x_exp, y_exp):
        return np.absolute(y_exp - self.model(p, x_exp))

    def chi2(self, p, x_exp, y_exp):
        return np.sum(self.residuals(p, x_exp, y_exp) ** 2)

    def on_stop_bt(self):
        r"""Function for stopping subprocesses."""
        self.stop_bt.configure(state=DISABLED)
        for i in self.process_list:
            if i.is_alive():
                i.terminate()
                i.join()
            else:
                print('No more alive')
        self.running_proc = [1] * self.num_fit
        self.stop_bt.configure(state=ACTIVE)
        self.start_bt.configure(state=ACTIVE)
        self.label.configure(text='Fit Canceled...')

    def on_start_bt(self):
        r"""Function for starting subprocesses."""
        self.running_proc = [0] * self.num_fit
        self.nbfits_x = []
        self.nbfits_errors = []
        self.line_fit.set_data([], [])
        self.line_fit2.set_data([], [])
        self.refresh()
        self.label.configure(text='Fitting...')
        self.start_bt.configure(state=DISABLED)
        self.stop_bt.configure(state=ACTIVE)

        for i in range(0, self.num_fit):
            worker = SimplexProcess(self.queue, 'Process: ' + str(i + 1),
                                    (self.x_exp, self.y_exp),
                                    self.p0,
                                    self.residuals,
                                    self.num_run)
            worker.daemon = True
            self.process_list.append(worker)
            worker.start()
        self.process_queue()

    def process_queue(self):
        r"""Function that is running every x ms for processing data pulled from subprocesses."""
        if sum(self.running_proc) < self.num_fit:
            try:
                outputs = self.queue.get(True)
                name, p_opt, error = outputs
                proc_index = int(name.split(':')[1]) - 1
                if p_opt != 'Done':
                    self.nbfits_x.append(error[0])
                    self.nbfits_errors.append(np.sum(error[1] ** 2))
                    self.display[proc_index].configure(text='Parameters: ' + str(p_opt))
                    if name == 'Process: 1':
                        self.line_error.set_data(self.nbfits_x, self.nbfits_errors)
                        self.line_fit.set_data(self.x_exp, self.model(p_opt, self.x_exp))
                        self.refresh()
                    elif name == 'Process: 2':
                        self.line_error.set_data(self.nbfits_x, self.nbfits_errors)
                        self.line_fit2.set_data(self.x_exp, self.model(p_opt, self.x_exp))
                        self.refresh()
                else:
                    self.running_proc[proc_index] = 1

                self.master.after(10, self.process_queue)

            except Queue.Empty:
                self.master.after(10, self.process_queue)

        else:
            self.label.configure(text='Fit Processed')
            self.start_bt.configure(state=ACTIVE)
            self.stop_bt.configure(state=DISABLED)
            print('All processes were processed')


class SimplexProcess(mp.Process):
    r"""Class documentation should be in __init__ method. PEP257."""
    def __init__(self, output_queue, name, inputs, p0, error_func, nb_run=3):
        r"""Multiprocess derivated class for starting a worker with a custom run function."""
        mp.Process.__init__(self)

        self.name = name
        self.output_queue = output_queue
        self.x_exp, self.y_exp = inputs
        self.p0 = p0
        self.func = error_func
        self.nb_run = nb_run
        self.fit_num = int(name.split(':')[1]) - 1

        self.p0 = np.absolute(np.random.randn(len(p0)))
        print(self.p0)

    def run(self):
        r"""Custom function ran in the subprocess."""
        print('Starting ' + self.name + '...')
        fopt = 0.0
        i = 0
        for i in range(self.nb_run):

            # xopt, fopt, iterations, funcalls, warnflag = optimize.fmin(self.func,
            #                                                           self.p0,
            #                                                           args=(self.x_exp, self.y_exp),
            #                                                           full_output=True,
            #                                                           retall=False,
            #                                                           disp=False,
            #                                                           callback=None)
            xopt, cov, infodict, msg, ier = optimize.leastsq(self.func, self.p0,
                                                             args=(self.x_exp, self.y_exp),
                                                             full_output=True)
            msg = (self.name, xopt, (i + self.fit_num * self.nb_run, infodict['fvec']))
            self.output_queue.put(msg)

            if xopt[0] < 0:
                xopt[0] = np.random.randn()

            if xopt[1] < 0:
                xopt[1] = np.random.randn()

            self.p0[:] = xopt[:]

        msg = (self.name, 'Done', (i + self.fit_num * self.nb_run, fopt))
        self.output_queue.put(msg)
        print('Exiting ' + self.name + '...')


if __name__ == '__main__':
    root = Tk()
    app = GUI(root)
    root.mainloop()
