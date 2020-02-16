#!/usr/bin/env python
__author__ = "James Schombert"
__credits__ = ["James Schombert"]
__email__ = "jschombe@uoregon.edu"

# This is a part of archangel photometry package ...
# See here for full details: http://abyss.uoregon.edu/~js/archangel/

import sys
import os
from tkinter import *
import Pmw


class listbox(Frame):
    def __init__(self, lines, parent=None):
        Frame.__init__(self, parent)
        Pmw.initialise(fontScheme='pmw1', size=16)

        w = Tk()
        try:
            self.tk.call('console', 'hide')
        except:
            pass
        self.s_width = w.winfo_screenwidth()
        self.s_height = w.winfo_screenheight()
        w.destroy()

        self.titles = []

        if '-f' in lines:
            self.titles = [tmp[:-1]
                           for tmp in open(lines[-1], 'r').readlines()]
            if '-x' in lines:
                os.system('mv -f '+lines[-1]+' ~/.Trash')
        else:
            while 1:
                try:
                    self.titles.append(input())
                except:
                    break

        self.text_window = 1

        self.tot_wid = 0
        self.box_height = min(len(self.titles), 30)
        for z in self.titles:
            if len(z) > self.tot_wid:
                self.tot_wid = len(z)
        if self.tot_wid > 80:
            self.tot_wid = 80
            self.tot_height = int((515-45)*self.box_height/30.)+60
        else:
            self.tot_height = int((495-45)*self.box_height/30.)+60
        self.tot_wid = int(9.*(self.tot_wid+2))

        geo = {'listbox': '+50+50'}
        master_geo = str(self.tot_wid)+'x'+str(self.tot_height)+geo['listbox']
        self.master.geometry(master_geo)
        self.butt_place = self.tot_height-43

        self.master.bind("<Up>", self.up_title)
        self.master.bind("<Down>", self.down_title)
        self.master.bind("<Return>", self.quit)

        self.frm = Frame()
        self.frm.configure(background='grey')
        self.button = Button(self.frm, text='Close',
                             bg='grey', command=self.cancel)
        self.button.place(x=0, y=self.butt_place,
                          width=self.tot_wid, height=35)
        self.frm.pack(side=BOTTOM, expand=YES, fill=BOTH, padx=5, pady=5)

        self.line = 0
        self.box()

    def box(self):
        self.listBox = Pmw.ScrolledListBox(self.frm,
                                           items=self.titles,
                                           listbox_height=self.box_height,
                                           vscrollmode="static",
                                           selectioncommand=self.select_line)
        self.listBox.component('listbox').configure(
            background='lavender', foreground='black', font='Courier 14')
        if not self.text_window:
            self.listBox.activate(self.line)
            self.listBox.select_set(ACTIVE)
            top, bottom = self.listBox.yview()
            size = bottom - top
        self.listBox.yview('moveto', float(self.line)/(1.5*len(self.titles)))
        self.listBox.place(x=0, y=0, width=self.tot_wid)
#      self.listBox.pack(fill=BOTH,padx=5,pady=5)

    def select_line(self):
        self.line = int(self.listBox.curselection()[0])
        self.box()

    def down_title(self, event):
        self.line = min(self.line+1, len(self.titles)-1)
        self.box()

    def up_title(self, event):
        self.line = max(self.line-1, 0)
        self.box()

    def cancel(self):
        self.master.destroy()

    def quit(self, event):
        self.out = self.titles[int(self.listBox.curselection()[0])]
        print(self.out)
        self.master.destroy()


if __name__ == '__main__':
    root = Tk()
    root.resizable(width=FALSE, height=FALSE)
    root.title('Help')
    root.configure(background='grey')
    listbox(sys.argv).mainloop()
