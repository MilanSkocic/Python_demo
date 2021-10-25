r"""
Very simple example for how the setup multiprocess with a tkinter app.
"""
import time, random
from tkinter import *
from tkinter.ttk import *
import multiprocessing as mp
import queue as Queue
from typing import Dict


class GUI(Frame):
    r"""Class documentation should be in __init__ method. PEP257."""
    def __init__(self, master: object, **kwargs: Dict):
        r"""Main tkinter app.

        Parameters
        ------------
        kwargs: dict, optional
            Tkinter frame kwargs.
        """
        super().__init__(master, **kwargs)
        self.master = master
        self.pack()
        self.label = Label(self, text='Process label')
        self.label.pack()
        self.start_bt = Button(self, text='Run Processes', command=self.on_start_bt)
        self.stop_bt = Button(self, text='Stop Processes', command=self.on_stop_bt)
        self.start_bt.pack()
        self.stop_bt.pack()
        self.process_list = list()
        self.queue = mp.SimpleQueue()
        self.running = False

    def _is_alive(self):

        flag = False
        for worker in self.process_list:
            flag += worker.is_alive()
        return flag
        
    def on_start_bt(self):
        r"""Function for starting subprocesses."""
        self.label.configure(text='Mutiprocessing...')
        for i in range(0, 3):
            worker = CustomProcess(self.queue, 'Process: ' + str(i + 1), 10)
            worker.daemon = True
            self.process_list.append(worker)
            worker.daemon = True
            worker.start()
            self.running = True
        self.process_queue()

    def process_queue(self):
        r"""Function that is running every x ms for processing data pulled from subprocesses."""
        if self.running:
            try:
                results, thread_name = self.queue.get()
                self.label.configure(text=str(results) + ' ' + str(thread_name))
                if results == "Done":
                    self.label.configure(text=f"{thread_name:s} finished.")
                    if self._is_alive():
                        self.master.after(100, self.process_queue)
                else:
                    self.master.after(100, self.process_queue)

            except Queue.Empty:
                self.master.after(100, self.process_queue)

    def on_stop_bt(self):
        r"""Function for stopping subprocesses."""
        for i in self.process_list:
            if i.is_alive():
                i.terminate()
                i.join()
        self.running = False
        self.label.configure(text='Fit Canceled...')

                    
class CustomProcess(mp.Process):
    r"""Class documentation should be in __init__ method. PEP257."""
    def __init__(self, queue, name, count):
        r"""Multiprocess derivated class for starting a worker with a custom run function."""
        mp.Process.__init__(self)
        self.name = name
        self.queue = queue
        self.count = count
                
    def run(self):
        r"""Custom function that is ran in the subprocess."""
        for i in range(0, self.count):
            time.sleep(random.random())
            self.queue.put((str(i), self.name))
        self.queue.put(('Done', self.name))


if __name__ == '__main__':
    mp.set_start_method("spawn")
    root = Tk()
    app = GUI(root)
    root.mainloop()
