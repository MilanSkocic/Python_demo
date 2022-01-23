r"""
Scrolled custom widgets.
"""
from tkinter import ttk
import tkinter as tk
class ScrolledFrame(ttk.Frame):
    r"""Class for scrolled frames. See __init__.__doc__."""

    def __init__(self, master, **kwargs):
        r"""
        Scrolled Frame widget which may contain other widgets and can have a 3D border.

        Parameters
        ------------
        master: tkinter widget
            Master container.
        kwargs: dict, optional
            Keyword arguments for the scrolled frame.
        """
        ttk.Frame.__init__(self, master)
        self._default_options = {'scrolled': 'y'}
        self.pack(expand=tk.TRUE, fill=tk.BOTH)
        for i in kwargs:
            if i not in self._default_options.keys():
                raise tk.TclError('Unknow option --' + i)

        self._default_options.update(kwargs)

        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self.yscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.xscrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL)

        if self._default_options['scrolled'] == 'y':
            self.yscrollbar.grid(row=0, column=1, sticky='ns')
        elif self._default_options['scrolled'] == 'x':
            self.xscrollbar.grid(row=1, column=0, sticky='ew')
        elif self._default_options['scrolled'] == 'both':
            self.yscrollbar.grid(row=0, column=1, sticky='ns')
            self.xscrollbar.grid(row=1, column=0, sticky='ew')
        else:
            raise tk.TclError('Bad scroll style \"' + 
                              self._default_options['scrolled'] + 
                              '\" must be x, y or both')

        self._canvas = tk.Canvas(self, bd=0, relief=tk.FLAT, 
                                 yscrollcommand=self.yscrollbar.set,
                                 xscrollcommand=self.xscrollbar.set)
        self._canvas.grid(row=0, column=0, sticky='nswe')

        self.yscrollbar.config(command=self._canvas.yview)
        self.xscrollbar.config(command=self._canvas.xview)

        self._canvas.config(scrollregion=self._canvas.bbox(tk.ALL))

        self._frame = ttk.Frame(self._canvas)
        self._frame.pack(expand=tk.TRUE, fill=tk.BOTH)
        self._frame.bind('<Configure>', self._update_canvas_window_size)

        self._canvas_window_id = self._canvas.create_window(0, 0, window=self._frame, anchor='nw')
        self._canvas.itemconfig(self._canvas_window_id, width=self._frame.winfo_reqwidth())
        self._canvas.bind("<Configure>", self._update_canvas_window_size)

    def _update_canvas_window_size(self, event):
        r"""Update canvas size when window is resized."""
        if event.width <= self._frame.winfo_reqwidth():
            self._canvas.itemconfig(self._canvas_window_id, width=self._frame.winfo_reqwidth())
        else:
            self._canvas.itemconfig(self._canvas_window_id, width=event.width)

        if event.height <= self._frame.winfo_reqheight():
            self._canvas.itemconfig(self._canvas_window_id, height=self._frame.winfo_reqheight())
        else:
            self._canvas.itemconfig(self._canvas_window_id, height=event.height)

        self._update_canvas_bbox()

    def _update_canvas_bbox(self):
        r"""Update scroll region when window is resized."""
        self._canvas.config(scrollregion=self._canvas.bbox(tk.ALL))

    @property
    def frame(self):
        r"""Return the frame that contains the widgets."""
        return self._frame

    @property
    def canvas(self):
        r"""Return the canvas that contains the scrollbars."""
        return self._canvas
