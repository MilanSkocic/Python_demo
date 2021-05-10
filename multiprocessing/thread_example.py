import threading
import Queue
from Tkinter import *
from ttk import *
import time


class GUI(Frame):

        def __init__(self, master):
        
                Frame.__init__(self, master)
                self.master = master
                self.pack()
                self.label = Label(self, text='Thread label')
                self.label.pack()
                self.labeltime = Label(self, text='%+06.1f' % 0.0)
                self.labeltime.pack()
                self.start_bt = Button(self, text='Run Threads', command=self.on_start_bt)
                self.start_bt.pack()
                self.stop_bt = Button(self, text='Stop Threads', command=self.on_stop_bt)
                self.stop_bt.pack()
                self.num_thread = 2
                self.thread_endflags = [0]*self.num_thread
                self.thread_list = []
                self.timer = 0.0
                print self.thread_endflags
        def on_start_bt(self):
                self.start_bt.configure(state=DISABLED)
                self.timer = 0.0
                self.labeltime.configure(text = '%+06.1f' % self.timer)
                self.thread_endflags = [0]*self.num_thread
                self.label.configure(text='MutliThread Starting...')
                self.queue = Queue.Queue()
                for i in range(0,self.num_thread):
                        worker = ThreadedTask(self.queue, 'Thread: ' + str(i+1),20)
                        worker.daemon = True
                        self.thread_list.append(worker)
                        worker.start()
                self.master.after(5, self.process_queue)
        def on_stop_bt(self):
                self.stop_bt.configure(state=DISABLED)
                for i in self.thread_list:
                        i.stop()
                        i.join()
                self.stop_bt.configure(state=ACTIVE)
        def process_queue(self):

                if sum(self.thread_endflags) < self.num_thread:
                        try:
                                results, thread_name = self.queue.get(False)
                                print('In process Queue: ' + thread_name + ' '+ str(results))
                                nb_thread = int(thread_name.split(':')[1])
                                if results == 'Done':
                                        self.thread_endflags[nb_thread-1]=1
                                        self.timer = self.timer + 0.3

                                        self.labeltime.configure(text = '%+06.1f' % self.timer)
                                        self.label.configure(text = thread_name + '-' + str(results))
                                        
                                        self.master.after(5, self.process_queue)
                                        
                                        
                                else:
                                        self.timer = self.timer + 0.3
                                                
                                        self.labeltime.configure(text = '%+06.1f' % self.timer)
                                        self.label.configure(text = thread_name + '-' + str(results))
                                                
                                        self.master.after(5, self.process_queue)
                                                         
                        except Queue.Empty:
                                self.timer = self.timer + 0.3
                                self.labeltime.configure(text = '%+06.1f' % self.timer)
                                self.master.after(5, self.process_queue)
                else:
                        self.start_bt.configure(state=NORMAL)
                        
        
class ThreadedTask(threading.Thread):
        def __init__(self, queue, name, count):
                super (ThreadedTask, self).__init__()
                self._stop = threading.Event()
                self.name = name
                self.queue = queue
                self.count = count
                
        def run(self):
                s = 0
                for i in range(0, self.count):
                        print self._stop.isSet()
                        if self._stop.isSet() == False:
                                print self._stop.isSet()
                                print('In ' + self.name + 'value: ' + str(i))
                                time.sleep(0.5)
                                self.queue.put((str(i), self.name))
                        else:
                                break
                self.queue.put(('Done' , self.name))
        def stop(self):
                self._stop.set() 
if __name__ == '__main__':
        root = Tk()
        app = GUI(root)
        root.mainloop()
