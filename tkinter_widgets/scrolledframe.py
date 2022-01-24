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
import tkinter as tk
from tkinter import ttk


class ScrolledFrame(ttk.Frame):
    r"""Class for scrolled frames. See __init__.__doc__."""

    def __init__(self, master, scrolled:str="both", **kwargs):
        r"""
        Scrolled Frame widget which may contain other widgets and can have a 3D border.

        Parameters
        ------------
        master: tkinter widget
            Master container.
        scrolled: str
            Indicates the scroll bars. Can be x, y, xy or both.
        kwargs: dict, optional
            Keyword arguments for the master Frame

        """
        ttk.Frame.__init__(self, master, **kwargs)
        self._default_options = ("x", "y", "xy", "both") 
        self.pack(expand=tk.TRUE, fill=tk.BOTH)
        if scrolled not in self._default_options:
            raise ValueError(f"scrolled can be x, y, xy or both.")


        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self.yscrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL)
        self.xscrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL)

        if scrolled == 'y':
            self.yscrollbar.grid(row=0, column=1, sticky='ns')
        elif scrolled == 'x':
            self.xscrollbar.grid(row=1, column=0, sticky='ew')
        elif (scrolled == 'both') or (scrolled == "xy"):
            self.yscrollbar.grid(row=0, column=1, sticky='ns')
            self.xscrollbar.grid(row=1, column=0, sticky='ew')
        else:
            raise ValueError(f"scrolled can be x, y, xy or both.")

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
