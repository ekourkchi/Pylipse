#!/usr/bin/python
__author__ = "Ehsan Kourkchi"
__copyright__ = "Copyright 2016"
__credits__ = ["Ehsan Kourkchi"]
__version__ = "1.0"
__maintainer__ = "Ehsan Kourkchi"
__email__ = "ehsan@ifa.hawaii.edu"
__status__ = "Production"


import sys
import os
import subprocess
import math
import numpy as np
from datetime import *
from pylab import *
import matplotlib
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import wcs
#from astropy import units as u
#from astropy.coordinates import SkyCoord
#from astropy.table import Table, Column

from mpl_toolkits.axes_grid1 import make_axes_locatable
from optparse import OptionParser
from PIL import Image, ImageTk

#import pyfits
import sys
import os
#import time
#import subprocess
#import signal
#from math import *
#from pylab import *
from matplotlib.ticker import MultipleLocator  # needed to fix up minor ticks
from matplotlib.patches import Ellipse, Polygon

#from scipy.stats import mode
########################


def xcmd(cmd):
    tmp = os.popen(cmd)
    output = ''
    for x in tmp:
        output += x
    if 'abort' in output:
        failure = True
    else:
        failure = tmp.close()
    if False:
        print('execution of %s failed' % cmd)
        print('error is as follows', output)
        sys.exit()
    else:
        return output

########################


def eplot(prf, ax, isophots=True):

    last_peek = 0
    prf_plot = 2

    if len(prf[0]) > 7:
        ells = [Ellipse((l[14], l[15]), 2.*l[3], 2.*l[3] *
                        (1.-l[12]), l[13], fill=0) for l in prf]
    else:
        #    for l in prf:
        #      print l[0],l[1],2.*(l[2]/(math.pi*(1.-l[3])))**0.5, \
        #            2.*(1.-l[3])*((l[2]/(math.pi*(1.-l[3])))**0.5),l[4]
        ells = [Ellipse((l[0], l[1]), 2.*(l[2]/(math.pi*(1.-l[3])))**0.5,
                        2.*(1.-l[3])*((l[2]/(math.pi*(1.-l[3])))**0.5), l[4], fill=0) for l in prf]

    last_e = None
    for e, l in zip(ells, prf):

        try:
            if l[6] == 0:          # no iterations or cleaned
                e.set_edgecolor('g')
                ax.add_artist(e)
                if not isophots:
                    e.set_visible(False)
            elif l[6] == -99:      # to be cleaned
                e.set_edgecolor('c')
                ax.add_artist(e)
                if isophots:
                    e.set_visible(False)
            elif l[6] == -1:       # iteration failed, used last ellipse
                e.set_edgecolor('r')
                ax.add_artist(e)
                if not isophots:
                    e.set_visible(False)
            else:
                # good fit
                e.set_clip_box(ax.bbox)
                e.set_alpha(1.0)
                e.set_edgecolor('b')
                last_e = e
                ax.add_artist(e)
                if not isophots:
                    e.set_visible(False)

        except:
            e.set_edgecolor('r')

        # if prf_plot == 2:
            # ax.plot([l[14]-(l[3]+5.)*cos(l[13]*pi/180.),l[14]+(l[3]+5.)*cos(l[13]*pi/180.)], \
            # [l[15]-(l[3]+5.)*sin(l[13]*pi/180.),l[15]+(l[3]+5.)*sin(l[13]*pi/180.)],'r')
            #bt=l[3]*(1.-l[12]) ; dt=((l[13]+90.)*pi/180.)
            # ax.plot([l[14]-(bt+5.)*cos(dt),l[14]+(bt+5.)*cos(dt)], \
            # [l[15]-(bt+5.)*sin(dt),l[15]+(bt+5.)*sin(dt)],'r')

    if last_e != None:
        # ax.add_artist(last_e)
        x, y = ells[0].center
        return x, y, last_e.width/2., last_e.height/2., last_e.angle, ells

########################


def read_ellipse(filename):

    prf = None   # ellipse data

    try:  # read xml file with xml_archangel classes
        doc = minidom.parse(filename.split('.')[0]+'.xml')
        rootNode = doc.documentElement
        elements = xml_read(rootNode).walk(rootNode)

        for t in elements['array']:
            if t[0]['name'] == 'prf':  # ellipse data
                prf = []
                head = []
                pts = []
                for z in t[2]['axis']:
                    head.append(z[0]['name'])
                    pts.append(list(map(float, z[1].split('\n'))))
                for z in range(len(pts[0])):
                    tmp = []
                    for w in head:
                        tmp.append(pts[head.index(w)][z])
# don't do this, causes small nummercal errors in iso_prf
#          if tmp[13] >= 270: tmp[13]=tmp[13]-360.
#          if tmp[13] > 90: tmp[13]=tmp[13]-180.
#          if tmp[13] <= -270: tmp[13]=tmp[13]+360.
#          if tmp[13] < -90: tmp[13]=tmp[13]+180.
                    prf.append(tmp)

    except:
        pass

    return prf
   ########################


def read_sky(filename):

    sky = None
    skysig = None

    try:  # read xml file with xml_archangel classes
        doc = minidom.parse(filename.split('.')[0]+'.xml')
        rootNode = doc.documentElement
        elements = xml_read(rootNode).walk(rootNode)

        for t in elements['sky']:
            if t[0]['units'] == 'DN':  # ellipse data
                sky = t[1]

        for t in elements['skysig']:
            if t[0]['units'] == 'DN':  # ellipse data
                skysig = t[1]

    except:
        pass

    return float(sky), float(skysig)


#################################################################
#################################################
def help():
    return '''
Version : v1.0 (March 2016)
email   : ehsan@ifa.hawaii.edu
Copyright 2016 Ehsan Kourkchi

********** Mouse Actions ***********
Use mouse middle wheel for scrolling
************************************

A) When mouse pointer is on the image

 a1) scroll-down  = zoom-in
 a2) scroll-up    = zoom-outer
 a3) middle-click = re-center the image

B) When anywhere on the GUI window
 
 b1) Ctrl+Scroll_up   = increase the semi-major axis
 b2) Ctrl+Scroll_down = decrease the semi-major axis
 
 b3) Shift+Scroll_up   = increase the semi-minor axis
 b4) Shift+Scroll-down = decrease the semi-minor axis

 b5) Alt+Scroll_up   = increase the position angle (PA)
 b6) Alt+Scroll_down = decrease the position angle (PA)
 
 i  ) PA increases in clock-wise direction. 
 ii ) PA = 0 if semi-major axis is horizontal
 iii) Step-size is displayed in the box next to the a-/b- and PA-control bars.
     - "Left  click" on the box: increases the step size by one pixel
     - "Right click" on the box: decreases the step size by one pixel
   
 b7) "Left/Right double click" = choose a new center for the ellipse
 b8) "Middle click" or "enter-key" = re-center the ellipse, if a new center has been already chosen
 b9) "Middle double click" = choose the new center and re-draw the ellipse at once (b7+b8)
 b10) z+Left_click = choose a new center for the ellipse (b7)
 b11) q/Esc = Ignoring the new chosen center 

C) When the fits file is displayed (no jpg file)
'c': chaning the contrast
'r': reset the contrst parameters

'''


#################################################

def arg_parser():
    parser = OptionParser(usage="""\

 - A GUI for manual ellipse fitting for Elliptical and Spiral galaxies ...

 - How to run: 
     %prog [options]

 - Use the -h option to see all options ...

 - You need to have at least a FITS file of your galaxy in either u,g,r,i,z bands. 
 - The file name must have this format: <object_name>_<filter>.fits
 - where filter is u, g, r, i, and/or z. 

 - You can also have these images in PNG format: 
    <object_name>_gri.png
    <object_name>_uri.png
    
  - In that case these images would be used for a colorful display. Otherwise the FITS file is used for the preview display.
  - This program us able to call DS9 program to open FITS files for manual ellipse fitting. 
  - FITS files wiil be also used for their WCS information.

  * To download ds9: http://ds9.si.edu/site/Download.html

  Note: PNG and FITS files should exactly match and have the same dimestions. 

 - Example: 
    $ python ellipse_fit.py -j pgc44182
      where: "pgc44182_g.fits" and "pgc44182_gri.png" are correscponding g-band FITS and PNG color images.
    $ python ellipse_fit.py -h 
      To see help and all available options.

""")
    parser.add_option('-j', '--object',
                      type='string', action='store',
                      help="""The object name""")

    (opts, args) = parser.parse_args()
    return opts, args
########


def ellipse_param(line):

    separator = []
    for i in range(len(line)):
        if line[i] == '(' or line[i] == ')':
            separator.append(i)
        elif line[i] == ',':
            separator.append(i)

    if len(separator) != 6:
        return None

    xcenter = line[separator[0]+1:separator[1]]
    ycenter = line[separator[1]+1:separator[2]]
    a = line[separator[2]+1:separator[3]]
    b = line[separator[3]+1:separator[4]]
    pa = line[separator[4]+1:separator[5]]

    xcenter = np.float(xcenter)
    ycenter = np.float(ycenter)
    a = np.float(a)
    b = np.float(b)
    pa = np.float(pa)

    state = [xcenter, ycenter, a, b, pa]

    return state


#################################################

def load_ellipse_ds9(reg_file):

    try:
        fo = open(reg_file, "rw+")
    except:
        print("\""+reg_file+"\": No such file ...")
        return None

    for i in range(20):

        line = fo.readline()
        if line[0:7] == 'ellipse':
            return ellipse_param(line)

    return None

#################################################
    #Circlx, Circly = myCircle(936, 1861, r)
    #circle, = ax.plot(Circlx, Circly, color='red', lw=1)
    # circle.set_dashes([2,3])


def myCircle(xcenter, ycenter, r):

    theta = np.arange(0.0, 360.0, 1.0)*np.pi/180.0
    Circlx = r*np.cos(theta) + xcenter
    Circly = r*np.sin(theta) + ycenter

    return Circlx, Circly

#################################################


def myEllipse(xcenter, ycenter, a, b, angle):

    theta = np.arange(0.0, 360.0, 1.0)*np.pi/180.0

    x = a * np.cos(theta)
    y = b * np.sin(theta)

    rtheta = np.radians(angle)
    R = np.array([
        [np.cos(rtheta), -np.sin(rtheta)],
        [np.sin(rtheta),  np.cos(rtheta)],
    ])

    x, y = np.dot(R, np.array([x, y]))
    x += xcenter
    y += ycenter

    return x, y

#################################################


class ImDisp:

    def __init__(self, Xmin, Xmax, Ymin, Ymax):
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.Ymin = Ymin
        self.Ymax = Ymax

        self.x1 = Xmin
        self.x2 = Xmax
        self.y1 = Ymin
        self.y2 = Ymax

    def zoom(self, xc, yc, ratio=1):

        delta_x = self.x2 - self.x1
        delta_y = self.y2 - self.y1

        dx = 0.5 * ratio * delta_x
        dy = 0.5 * ratio * delta_y

        ###
        if xc + dx > self.Xmax:
            self.x2 = self.Xmax
            self.x1 = self.Xmax - 2. * dx
        elif xc - dx < self.Xmin:
            self.x1 = self.Xmin
            self.x2 = self.Xmin + 2. * dx
        else:
            self.x1 = xc - dx
            self.x2 = xc + dx

        if self.x1 < self.Xmin:
            self.x1 = self.Xmin
        if self.x2 > self.Xmax:
            self.x2 = self.Xmax
        ###
        if yc + dy > self.Ymax:
            self.y2 = self.Ymax
            self.y1 = self.Ymax - 2. * dy
        elif yc - dy < self.Ymin:
            self.y1 = self.Ymin
            self.y2 = self.Ymin + 2. * dy
        else:
            self.y1 = yc - dy
            self.y2 = yc + dy

        if self.y1 < self.Ymin:
            self.y1 = self.Ymin
        if self.y2 > self.Ymax:
            self.y2 = self.Ymax
        ###

    def zoom_IN(self, xc, yc, ratio=0.75):
        self.zoom(xc, yc, ratio=ratio)
        return self.x1, self.x2, self.y1, self.y2

    def zoom_OUT(self, xc, yc, ratio=4./3):
        self.zoom(xc, yc, ratio=ratio)
        return self.x1, self.x2, self.y1, self.y2

    def reset(self):
        self.x1 = self.Xmin
        self.x2 = self.Xmax
        self.y1 = self.Ymin
        self.y2 = self.Ymax
        return self.x1, self.x2, self.y1, self.y2

    def pan(self, xc, yc):
        self.zoom(xc, yc, ratio=1)
        return self.x1, self.x2, self.y1, self.y2


#################################################
class Undo_Redo:

    def __init__(self):

        self.data = []
        self.n = 0
        self.iter = 0
        self.size = 100

    def push(self, item):
        self.data = self.data[0:self.iter]
        self.iter += 1
        self.data.append(item)
        self.n = self.iter

        if self.n > self.size:
            self.data = self.data[self.n-self.size:self.n]
            self.n = self.size
            self.iter = self.size

    def undo(self):
        if self.iter > 1:
            self.iter -= 1
            return self.data[self.iter-1]

        return None

    def redo(self):
        if self.iter < self.n:
            self.iter += 1
            return self.data[self.iter-1]
        return None

    def current(self):
        return self.data[self.iter-1]
#################################################


class objEllipse():

    def __init__(self, ax, xcenter, ycenter, a, b, pa, color='red'):

        self.xcenter = xcenter
        self.ycenter = ycenter
        self.a = a
        self.b = b
        self.pa = pa

        ellX, ellY = myEllipse(xcenter, ycenter, a, b, pa)
        self.ellipse, = ax.plot(ellX, ellY, color=color, lw=1)
        self.ellipse_c, = ax.plot(xcenter, ycenter, 'r+')
        self.ellipse.set_dashes([2, 3])

        annotate("Center ", (0.016, 0.42),
                 xycoords='figure fraction', size=14, color='maroon')

        self.uXc = annotate("Xc: "+'{:.1f}'.format(xcenter),
                            (0.015, 0.39), xycoords='figure fraction', size=10)
        self.uYc = annotate("Yc: "+'{:.1f}'.format(ycenter),
                            (0.015, 0.36), xycoords='figure fraction', size=10)

        state = [xcenter, ycenter, a, b, pa]
        self.state_iter = Undo_Redo()
        self.state_iter.push(state)

        b_a = float(b)/float(a)
        self.b_a = annotate(
            "b/a: "+'{:.2f}'.format(b_a), (0.20, 0.20), xycoords='figure fraction', size=12)

    def update(self, xcenter, ycenter, a, b, pa):

        self.xcenter = np.float(xcenter)
        self.ycenter = np.float(ycenter)
        self.a = np.float(a)
        self.b = np.float(b)
        self.pa = np.float(pa)

        ellX, ellY = myEllipse(xcenter, ycenter, a, b, pa)
        self.ellipse.set_xdata(ellX)
        self.ellipse.set_ydata(ellY)
        self.ellipse_c.set_xdata([xcenter])
        self.ellipse_c.set_ydata([ycenter])

        self.uXc.set_text("Xc: "+'{:.1f}'.format(xcenter))
        self.uYc.set_text("Yc: "+'{:.1f}'.format(ycenter))

        state = [xcenter, ycenter, a, b, pa]
        self.state_iter.push(state)
        b_a = float(b)/float(a)
        self.b_a.set_text("b/a: "+'{:.2f}'.format(b_a))

        draw()

    def undo_redo(self, undo=True):

        if undo == True:
            state = self.state_iter.undo()
        else:
            state = self.state_iter.redo()

        if state != None:
            [xcenter, ycenter, a, b, pa] = state

            self.xcenter = xcenter
            self.ycenter = ycenter
            self.a = a
            self.b = b
            self.pa = pa

            ellX, ellY = myEllipse(xcenter, ycenter, a, b, pa)
            self.ellipse.set_xdata(ellX)
            self.ellipse.set_ydata(ellY)
            self.ellipse_c.set_xdata([xcenter])
            self.ellipse_c.set_ydata([ycenter])
            self.uXc.set_text("Xc: "+'{:.1f}'.format(xcenter))
            self.uYc.set_text("Yc: "+'{:.1f}'.format(ycenter))
            b_a = float(b)/float(a)
            self.b_a.set_text("b/a: "+'{:.2f}'.format(b_a))
            draw()
        return state

    def update_color(self, color):
        self.ellipse.set_color(color)
        self.ellipse_c.set_color(color)


has_counter = False

a = 443
b = 208
pa = 23
xcenter = 1361
ycenter = 1367

delta_axis = 10
delta_angle = 5
delta_position = 10

x_new = 0
y_new = 0
center_change = False

plus_minus = r'$\pm$'

plotFits = False
r1 = 0
r1_org = 0
r2 = 0
r2_org = 0
middle = 0
mid_org = 0
imgplot = None

inv = -1


def main(root_name):
    global a, b, pa, xcenter, ycenter, plotFits, imgplot, r1, r2, r1_org, r2_org, middle, mid_org, inv

    u_file = root_name+'_u.fits'
    g_file = root_name+'_g.fits'
    r_file = root_name+'_r.fits'
    i_file = root_name+'_i.fits'
    z_file = root_name+'_z.fits'
    HaveFitsFile = True
    plotFits = False

    gri_image = None
    urz_image = None

    has_u_fits = os.path.isfile(u_file)
    has_g_fits = os.path.isfile(g_file)
    has_r_fits = os.path.isfile(r_file)
    has_i_fits = os.path.isfile(i_file)
    has_z_fits = os.path.isfile(z_file)

    jpeg_gri = root_name+'_gri.jpg'
    jpeg_urz = root_name+'_urz.jpg'

    fig = plt.figure(figsize=(10, 8), dpi=100)
    ax = fig.add_axes([0.2, 0.25, 0.7,  0.70])
    subplots_adjust(left=0.25, bottom=0.25)
    fig.patch.set_facecolor('lightgray')

    if has_g_fits == False:
        print(g_file + '  does not exist ... !!! \n')
        if has_r_fits:
            fits_file = r_file
        elif has_i_fits:
            fits_file = i_file
        elif has_z_fits:
            fits_file = z_file
        elif has_u_fits:
            fits_file = u_file
        else:
            HaveFitsFile = False
    else:
        fits_file = root_name+'_g.fits'

    if HaveFitsFile:
        hdu_list = fits.open(fits_file)
        image_data = hdu_list[0].data
        w = wcs.WCS(hdu_list[0].header)
        fits_x_max, fits_y_max = image_data.shape

    if os.path.isfile(jpeg_gri):
        img = Image.open(jpeg_gri)
        rsize = img.resize((img.size[0], img.size[1]))  # Use PIL to resize
        rsizeArr = np.asarray(rsize)  # Get array back
        # rsizeArr = np.fliplr(rsizeArr) # left/right flip
        gri_image = np.flipud(rsizeArr)  # up/down flip
        gri_x_max, gri_y_max, dimension = gri_image.shape
    else:
        gri_image = None

    if os.path.isfile(jpeg_urz):
        img = Image.open(jpeg_urz)
        rsize = img.resize((img.size[0], img.size[1]))  # Use PIL to resize
        rsizeArr = np.asarray(rsize)  # Get array back
        urz_image = np.flipud(rsizeArr)  # up/down flip
        urz_x_max, urz_y_max, dimension = urz_image.shape
    else:
        urz_image = None

    if gri_image is not None:
        imgplot = plt.imshow(gri_image)
        x_max = gri_x_max
        y_max = gri_y_max
        x_min = 0
        y_min = 0
        axis([x_min, x_max, y_min, y_max])
    elif urz_image is not None:
        imgplot = plt.imshow(urz_image)
        x_max = urz_x_max
        y_max = urz_y_max
        x_min = 0
        y_min = 0
        axis([x_min, x_max, y_min, y_max])
    elif HaveFitsFile:
        plotFits = True
        r2 = np.min(image_data)
        r1 = np.max(image_data)

        try:
            xsky, skysig = read_sky(root_name+'_g.xml')
            if xsky != None:
                r2 = 0.7*xsky
                r1 = 3.*xsky
                middle = (r1-r2)/2.
                sig = 50.
                r1 = xsky+sig*skysig
                r2 = xsky-0.05*(r1-xsky)
                middle = xsky
        except:
            pass

        r2_org = r2
        r1_org = r1
        mid_org = middle

        x_max = fits_x_max
        y_max = fits_y_max
        x_min = 0
        y_min = 0
        palette = cm.gray
        palette.set_bad('r', 1.0)
        imgplot = plt.imshow(inv*image_data, interpolation='nearest', cmap=palette,
                             aspect='equal', origin='lower', vmin=min(inv*r1, inv*r2), vmax=max(inv*r2, inv*r1))  # , norm=LogNorm())
        axis([x_min, x_max, y_min, y_max])
    else:
        print("Nither JPG file nor FITS file ... !!!!")
        sys.exit(0)

    disp = ImDisp(x_min, x_max, y_min, y_max)

    save_name = root_name+'_ellipse.txt'
    if os.path.isfile(save_name):
        el = np.genfromtxt(save_name, delimiter=',',
                           filling_values="-100000", names=True, dtype=np.float32)
        xcenter = el['xcenter']
        ycenter = el['ycenter']
        a = el['a_pix']
        b = el['b_pix']
        pa = el['pa_deg']
        if xcenter.size > 1:
            xcenter = np.float(xcenter[0])
            ycenter = np.float(ycenter[0])
            a = np.float(a[0])
            b = np.float(b[0])
            pa = np.float(pa[0])
    else:
        xcenter = np.round(x_max/2.)
        ycenter = np.round(y_max/2.)
        a = xcenter/5.
        b = a/2.
        pa = 45.

    semi_a_max = np.round(0.75 * x_max)
    semi_b_max = np.round(0.75 * y_max)

    resetax = axes([0.01, 0.34, 0.18, 0.43])
    info_bts = Button(resetax, '', color='lightgray', hovercolor='lightgray')

    resetax = axes([0.86, 0.65, 0.13, 0.31])
    ds9_bts = Button(resetax, '', color='lightgray', hovercolor='lightgray')

    annotate("object: " + root_name, (0.45, 0.02),
             xycoords='figure fraction', size=12, color='maroon')


    user_center, = ax.plot([], [],  'ro')

    isophots = None
    if os.path.isfile(root_name+'_g.xml'):
        prf = read_ellipse(root_name+'_g.xml')
        xcenter_iso, ycenter_iso, a_iso, b_iso, pa_iso, isophots = eplot(
            prf, ax, isophots=True)  # if True, plots right away
        if not os.path.isfile(save_name):
            xcenter = xcenter_iso
            ycenter = ycenter_iso
            a = a_iso
            b = b_iso
            pa = pa_iso

    ellipse = objEllipse(ax, np.float(xcenter), np.float(
        ycenter), np.float(a), np.float(b), np.float(pa))

    axcolor = 'lightgoldenrodyellow'

    #######################

    ax_a = axes([0.22, 0.15, 0.60, 0.02])
    ax_b = axes([0.22, 0.1, 0.60, 0.02])
    ax_pa = axes([0.22, 0.05, 0.60, 0.02])

    slider_a = Slider(ax_a, 'a', 0, semi_a_max, valinit=a, dragging=True)
    slider_b = Slider(ax_b, 'b', 0, semi_b_max, valinit=b, dragging=True)
    slider_pa = Slider(ax_pa, 'PA', 0, 180, valinit=pa, dragging=True)

    def update(val):
        global a, b, pa, xcenter, ycenter
        a = slider_a.val
        b = slider_b.val
        pa = slider_pa.val
        ellipse.update(np.float(xcenter), np.float(ycenter),
                       np.float(a), np.float(b), np.float(pa))

    slider_a.on_changed(update)
    slider_b.on_changed(update)
    slider_pa.on_changed(update)
    #######################
    #######################
    rax = axes([0.020, 0.045, 0.15, 0.15])
    radio = RadioButtons(rax, ('red', 'blue', 'lawngreen', 'yellow'), active=0)
    annotate('ellipse color', (0.05, 0.025),
             xycoords='figure fraction', size=8, color='black')

    def colorfunc(label):
        ellipse.update_color(label)
        draw()

    radio.on_clicked(colorfunc)
    #######################
    #######################
    if (gri_image is None or urz_image is None) and isophots is not None:
        iso_image = axes([0.02, 0.22, 0.15, 0.08])
        radio_isoimage = RadioButtons(iso_image, ('on', 'off'), active=0)
        annotate('ellipse-isophot', (0.05, 0.225),
                 xycoords='figure fraction', size=8, color='black')

    def iso_onoff(status):
        if status == 'on':
            for ellipse in isophots:
                ellipse.set_visible(True)
        elif status == 'off':
            for ellipse in isophots:
                ellipse.set_visible(False)

    if isophots is not None:
        radio_isoimage.on_clicked(iso_onoff)
    #######################
    #######################

    if gri_image is not None and urz_image is not None:
        rax_image = axes([0.02, 0.22, 0.15, 0.08])
        radio_image = RadioButtons(rax_image, ('gri', 'urz'), active=0)

    def change_image(color):
        if color == 'gri':
            imgplot = ax.imshow(gri_image)
            draw()
        elif color == 'urz':
            imgplot = ax.imshow(urz_image)
            draw()

    if gri_image is not None and urz_image is not None:
        radio_image.on_clicked(change_image)
    #######################

    resetax = axes([0.93, 0.125, 0.05, 0.05])
    delt_axis_button = Button(
        resetax, plus_minus+str(delta_axis), color=axcolor, hovercolor='navajowhite')

    def addOne_axis(event):
        global delta_axis
        if event.button == 1:
            delta_axis += 1
            delt_axis_button.label.set_text(plus_minus+str(delta_axis))
            draw()
        elif event.button == 3:
            delta_axis -= 1
            delt_axis_button.label.set_text(plus_minus+str(delta_axis))
            draw()

    delt_axis_button.on_clicked(addOne_axis)
    #######################
    tmp = Image.open('./icons/Help_icon.png')
    help_label = annotate("HELP", (0.035, 0.87),
                          xycoords='figure fraction', size=12)
    tmp_rsize = tmp.resize((tmp.size[0], tmp.size[1]))
    help_icon = np.asarray(tmp_rsize)
    #help_icon = np.flipud(help_icon)

    resetax = axes([0.02, 0.9, 0.07, 0.07])
    help_button = Button(resetax, '?', color='white',
                         hovercolor='yellow', image=help_icon)

    def help_me(event):
        fout = open('help.tmp', 'w')
        fout.write(help()+'\n')
        fout.close()
        subprocess.Popen(
            'python /home/ehsan/PanStarrs/glga/data.old/194D/sdss/jpg/help_widget.py  -x -f help.tmp &', shell=True)

    help_button.on_clicked(help_me)
    #######################
    #######################
    tmp = Image.open('./icons/home_icon.png')
    home_label = annotate("Home", (0.115, 0.87),
                          xycoords='figure fraction', size=12)
    tmp_rsize = tmp.resize((tmp.size[0], tmp.size[1]))
    home_icon = np.asarray(tmp_rsize)
    #home_icon = np.flipud(home_icon)

    resetax = axes([0.1, 0.9, 0.07, 0.07])
    home_button = Button(resetax, ' ', color='white',
                         hovercolor='yellow', image=home_icon)

    def home_disp(event):
        i1, i2, j1, j2 = disp.reset()
        ax.set_xlim(i1, i2)
        ax.set_ylim(j1, j2)
        draw()

    home_button.on_clicked(home_disp)
    #######################
    resetax = axes([0.86, 0.25, 0.055, 0.055])

    undo_button = Button(resetax, 'Undo', color='tan',
                         hovercolor='navajowhite')

    def undo(event):
        global a, b, pa, xcenter, ycenter
        state = ellipse.undo_redo(undo=True)
        if state != None:
            [xcenter, ycenter, a, b, pa] = state

            slider_a.disconnect(0)
            slider_a.set_val(a)
            slider_a.observers[0] = update

            slider_b.disconnect(0)
            slider_b.set_val(b)
            slider_b.observers[0] = update

            slider_pa.disconnect(0)
            slider_pa.set_val(pa)
            slider_pa.observers[0] = update

    undo_button.on_clicked(undo)
    #######################
    #######################

    resetax = axes([0.93, 0.25, 0.055, 0.055])
    redo_button = Button(resetax, 'Redo', color='tan',
                         hovercolor='navajowhite')

    def redo(event):
        global a, b, pa, xcenter, ycenter
        state = ellipse.undo_redo(undo=False)
        if state != None:
            [xcenter, ycenter, a, b, pa] = state

            slider_a.disconnect(0)
            slider_a.set_val(a)
            slider_a.observers[0] = update

            slider_b.disconnect(0)
            slider_b.set_val(b)
            slider_b.observers[0] = update

            slider_pa.disconnect(0)
            slider_pa.set_val(pa)
            slider_pa.observers[0] = update

    redo_button.on_clicked(redo)
    #######################
    tmp = Image.open('./icons/Save-icon.png')
    tmp_rsize = tmp.resize((tmp.size[0], tmp.size[1]))
    save_icon = np.asarray(tmp_rsize)
    #save_icon = np.flipud(save_icon)
    resetax = axes([0.92, 0.35, 0.07, 0.07])
    annotate("Save", (0.865, 0.39), xycoords='figure fraction', size=10)
    annotate("Ellipse", (0.86, 0.365), xycoords='figure fraction', size=10)

    save_button = Button(resetax, '', color='white',
                         hovercolor='navajowhite', image=save_icon)

    def save(event):
        global a, b, pa, xcenter, ycenter

        save_name = root_name+'_ellipse.txt'

        world = w.wcs_pix2world([[xcenter, ycenter]], 1)
        RA = world[0][0]
        DEC = world[0][1]

        ellipse_string = "ellipse, " + '{:.5f}'.format(RA) + ', ' + '{:.5f}'.format(DEC) + ', ' + '{:.2f}'.format(
            xcenter)+', '+'{:.2f}'.format(ycenter)+', '+'{:.2f}'.format(a)+', '+'{:.2f}'.format(b)+', '+'{:.2f}'.format(pa)
        ellipse_string += datetime.datetime.now().strftime(', %b-%d-%Y, %H:%M:%S')

        header = "ellipse, RA_deg, DEC_deg, xcenter, ycenter, a_pix, b_pix, pa_deg, date, time \n"

        if os.path.isfile(save_name):
            with open(save_name, "r+") as f:
                old = f.read()  # read everything in the file
                l_old = len(old)
                i = 0
                while i < l_old:
                    if old[i] == '\n':
                        break
                    i += 1

                if i+1 < l_old:
                    old = old[i+1:l_old]
                f.seek(0)  # rewind
                # write the new line before
                f.write(header + ellipse_string+"\n" + old)

        else:
            with open(save_name, "w+") as f:
                f.write(header + ellipse_string+"\n")

    save_button.on_clicked(save)

    #######################

    ds9_label = annotate("DS9", (0.94, 0.92),
                         xycoords='figure fraction', size=12)

    resetax = axes([0.87, 0.90, 0.05, 0.05])
    if has_u_fits:
        ds9_u_button = Button(
            resetax, 'u', color='lightblue', hovercolor='navajowhite')
    else:
        ds9_u_button = Button(resetax, 'u', color='grey', hovercolor='grey')

    def ds9_u_open(event):
        global a, b, pa, xcenter, ycenter

        ds9_command = "ds9 " + u_file
        ds9_command += " -width 500"
        ds9_command += " -scale log"
        ds9_command += " -scale minmax"
        ds9_command += " -zoom to fit"
        ds9_command += " -pan to "+str(xcenter)+" "+str(ycenter)

        os.system(ds9_command+" &")

        os.system("sleep 5")

        os.system("xpaset -p ds9 regions command \"{ellipse "+str(xcenter)+" "+str(
            ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2}\"")
        os.system("xpaset -p ds9 regions system  image")
        os.system("xpaset -p ds9 regions format ds9")
        os.system("xpaset -p ds9 regions save foo.reg")

    if has_u_fits:
        ds9_u_button.on_clicked(ds9_u_open)
    #######################
    #######################
    #######################

    resetax = axes([0.87, 0.83, 0.05, 0.05])
    if has_g_fits:
        ds9_g_button = Button(
            resetax, 'g', color='lightgreen', hovercolor='navajowhite')
    else:
        ds9_g_button = Button(resetax, 'g', color='grey', hovercolor='grey')

    def ds9_g_open(event):
        global a, b, pa, xcenter, ycenter

        ds9_command = "ds9 " + g_file
        ds9_command += " -height 500"
        ds9_command += " -width 500"
        ds9_command += " -scale log"
        ds9_command += " -scale minmax"
        ds9_command += " -zoom to fit"
        ds9_command += " -pan to "+str(xcenter)+" "+str(ycenter)

        os.system(ds9_command+" &")

        os.system("sleep 5")

        os.system("xpaset -p ds9 regions command \"{ellipse "+str(xcenter)+" "+str(
            ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2}\"")
        os.system("xpaset -p ds9 regions system  image")
        os.system("xpaset -p ds9 regions format ds9")
        os.system("xpaset -p ds9 regions save foo.reg")

    if has_g_fits:
        ds9_g_button.on_clicked(ds9_g_open)
    #######################
    #######################

    resetax = axes([0.93, 0.83, 0.05, 0.05])
    if has_r_fits:
        ds9_r_button = Button(resetax, 'r', color='red',
                              hovercolor='navajowhite')
    else:
        ds9_r_button = Button(resetax, 'r', color='grey', hovercolor='grey')

    def ds9_r_open(event):
        global a, b, pa, xcenter, ycenter

        ds9_command = "ds9 " + r_file
        ds9_command += " -height 500"
        ds9_command += " -width 500"
        ds9_command += " -scale log"
        ds9_command += " -scale minmax"
        ds9_command += " -zoom to fit"
        ds9_command += " -pan to "+str(xcenter)+" "+str(ycenter)
        # ds9_command += " -regions command \"ellipse "+str(xcenter)+" "+str(ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2\""
        os.system(ds9_command+" &")

        os.system("sleep 3")
        ##os.system("xpaset -p ds9 pan to "+str(xcenter)+" "+str(ycenter))
        #os.system("xpaset -p ds9 scale minmax")
        #os.system("xpaset -p ds9 scale log")
        #os.system("xpaset -p ds9 height 500")
        #os.system("xpaset -p ds9 width 500")
        os.system("xpaset -p ds9 regions command \"{ellipse "+str(xcenter)+" "+str(
            ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2}\"")
        #os.system("xpaset -p ds9 zoom to fit")
        os.system("xpaset -p ds9 regions system  image")
        os.system("xpaset -p ds9 regions format ds9")
        os.system("xpaset -p ds9 regions save foo.reg")

    if has_r_fits:
        ds9_r_button.on_clicked(ds9_r_open)
    #######################
    #######################
    #######################

    resetax = axes([0.87, 0.76, 0.05, 0.05])
    if has_i_fits:
        ds9_i_button = Button(resetax, 'i', color='orange',
                              hovercolor='navajowhite')
    else:
        ds9_i_button = Button(resetax, 'i', color='grey', hovercolor='grey')

    def ds9_i_open(event):
        global a, b, pa, xcenter, ycenter

        ds9_command = "ds9 " + i_file
        ds9_command += " -height 500"
        ds9_command += " -width 500"
        ds9_command += " -scale log"
        ds9_command += " -scale minmax"
        ds9_command += " -zoom to fit"
        ds9_command += " -pan to "+str(xcenter)+" "+str(ycenter)
        # ds9_command += " -regions command \"ellipse "+str(xcenter)+" "+str(ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2\""
        os.system(ds9_command+" &")

        os.system("sleep 3")
        ##os.system("xpaset -p ds9 pan to "+str(xcenter)+" "+str(ycenter))
        #os.system("xpaset -p ds9 scale minmax")
        #os.system("xpaset -p ds9 scale log")
        #os.system("xpaset -p ds9 height 500")
        #os.system("xpaset -p ds9 width 500")
        os.system("xpaset -p ds9 regions command \"{ellipse "+str(xcenter)+" "+str(
            ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2}\"")
        #os.system("xpaset -p ds9 zoom to fit")
        os.system("xpaset -p ds9 regions system  image")
        os.system("xpaset -p ds9 regions format ds9")
        os.system("xpaset -p ds9 regions save foo.reg")

    if has_i_fits:
        ds9_i_button.on_clicked(ds9_i_open)
    #######################
    #######################

    resetax = axes([0.93, 0.76, 0.05, 0.05])
    if has_z_fits:
        ds9_z_button = Button(resetax, 'z', color='gold',
                              hovercolor='navajowhite')
    else:
        ds9_z_button = Button(resetax, 'z', color='grey', hovercolor='grey')

    def ds9_z_open(event):
        global a, b, pa, xcenter, ycenter

        ds9_command = "ds9 " + z_file
        ds9_command += " -height 500"
        ds9_command += " -width 500"
        ds9_command += " -scale log"
        ds9_command += " -scale minmax"
        ds9_command += " -zoom to fit"
        ds9_command += " -pan to "+str(xcenter)+" "+str(ycenter)
        # ds9_command += " -regions command \"ellipse "+str(xcenter)+" "+str(ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2\""
        os.system(ds9_command+" &")

        os.system("sleep 3")
        ##os.system("xpaset -p ds9 pan to "+str(xcenter)+" "+str(ycenter))
        #os.system("xpaset -p ds9 scale minmax")
        #os.system("xpaset -p ds9 scale log")
        #os.system("xpaset -p ds9 height 500")
        #os.system("xpaset -p ds9 width 500")
        os.system("xpaset -p ds9 regions command \"{ellipse "+str(xcenter)+" "+str(
            ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2}\"")
        #os.system("xpaset -p ds9 zoom to fit")
        os.system("xpaset -p ds9 regions system  image")
        os.system("xpaset -p ds9 regions format ds9")
        os.system("xpaset -p ds9 regions save foo.reg")

    if has_z_fits:
        ds9_z_button.on_clicked(ds9_z_open)
    #######################

    resetax = axes([0.92, 0.55, 0.07, 0.07])
    annotate("Export", (0.86, 0.59), xycoords='figure fraction', size=10)
    annotate("to ds9", (0.865, 0.565), xycoords='figure fraction', size=8)

    tmp = Image.open('./icons/right_arrow.png')
    tmp_rsize = tmp.resize((tmp.size[0], tmp.size[1]))
    right_arrow = np.asarray(tmp_rsize)
    ds9_sync_button = Button(
        resetax, '', color='cornsilk', hovercolor='navajowhite', image=right_arrow)
    ds9_sync_button.label.set_fontsize(10)

    def ds9_sync(event):
        global a, b, pa, xcenter, ycenter

        os.system("xpaset -p ds9 regions delete all")
        os.system("xpaset -p ds9 regions command \"{ellipse "+str(xcenter)+" "+str(
            ycenter)+" "+str(a)+" "+str(b)+" "+str(pa)+" # color=red width=2}\"")
        os.system("xpaset -p ds9 regions system  image")
        os.system("xpaset -p ds9 regions format ds9")
        os.system("xpaset -p ds9 regions save foo.reg")

    ds9_sync_button.on_clicked(ds9_sync)
    #######################
    resetax = axes([0.92, 0.45, 0.07, 0.07])
    annotate("Import", (0.86, 0.49), xycoords='figure fraction', size=10)
    annotate("from ds9", (0.86, 0.465), xycoords='figure fraction', size=8)

    tmp = Image.open('./icons/left_arrow.png')
    tmp_rsize = tmp.resize((tmp.size[0], tmp.size[1]))
    left_arrow = np.asarray(tmp_rsize)
    esn_sync_button = Button(
        resetax, '', color='cornsilk', hovercolor='navajowhite', image=left_arrow)
    esn_sync_button.label.set_fontsize(10)

    def ds9_sync(event):
        global a, b, pa, xcenter, ycenter

        os.system("xpaset -p ds9 regions system  image")
        os.system("xpaset -p ds9 regions format ds9")
        os.system("xpaset -p ds9 regions save foo.reg")
        state = load_ellipse_ds9("foo.reg")
        if state != None:
            #print state
            [xcenter, ycenter, a, b, pa] = state

            ellipse.update(np.float(xcenter), np.float(ycenter),
                           np.float(a), np.float(b), np.float(pa))

            slider_a.disconnect(0)
            slider_a.set_val(a)
            slider_a.observers[0] = update

            slider_b.disconnect(0)
            slider_b.set_val(b)
            slider_b.observers[0] = update

            slider_pa.disconnect(0)
            slider_pa.set_val(pa)
            slider_pa.observers[0] = update

    esn_sync_button.on_clicked(ds9_sync)
    #######################
    resetax = axes([0.89, 0.67, 0.07, 0.07])
    ds9_contour_button = Button(
        resetax, 'Contours\nOn', color='lightgrey', hovercolor='navajowhite')
    ds9_contour_button.label.set_fontsize(10)

    def ds9_counter(event):
        global has_counter

        if has_counter == False:
            os.system("xpaset -p ds9 contour nlevels 30")
            os.system("xpaset -p ds9 contour smooth 10")
            os.system("xpaset -p ds9 contour scale log")
            os.system("xpaset -p ds9 contour generate")
            os.system("xpaset -p ds9 contour color green")
            os.system("xpaset -p ds9 contour")

            ds9_contour_button.label.set_text('Contours\nOff')
            ds9_contour_button.color = 'cornsilk'
            has_counter = True
        else:
            os.system("xpaset -p ds9 contour clear")
            ds9_contour_button.label.set_text('Contours\nOn')
            ds9_contour_button.color = 'lightgrey'
            has_counter = False

    ds9_contour_button.on_clicked(ds9_counter)
    #######################

    #######################

    resetax = axes([0.93, 0.045, 0.05, 0.05])
    delt_angle_button = Button(
        resetax, plus_minus+str(delta_angle), color='tan', hovercolor='navajowhite')

    def addOne_angle(event):
        global delta_angle
        if event.button == 1:
            delta_angle += 1
            delt_angle_button.label.set_text(plus_minus+str(delta_angle))
            draw()
        elif event.button == 3:
            delta_angle -= 1
            delt_angle_button.label.set_text(plus_minus+str(delta_angle))
            draw()

    delt_angle_button.on_clicked(addOne_angle)
    #######################
    #######################

    resetax = axes([0.12, 0.355, 0.05, 0.05])
    delt_position_button = Button(
        resetax, plus_minus+str(delta_position), color='gainsboro', hovercolor='navajowhite')

    def addOne_position(event):
        global delta_position
        if event.button == 1:
            delta_position += 1
            delt_position_button.label.set_text(plus_minus+str(delta_position))
            draw()
        elif event.button == 3:
            delta_position -= 1
            delt_position_button.label.set_text(plus_minus+str(delta_position))
            draw()

    delt_position_button.on_clicked(addOne_position)
    #######################

    #######################

    def scroll_event(event):
        #print 'you pressed', event.key, event.button, event.xdata, event.ydata, event.key

        global a, b, pa, xcenter, ycenter, delta_axis, delta_angle

        if event.inaxes == ax:
            if event.key is None and event.button == 'up':
                #print "Zoom out"
                i1, i2, j1, j2 = disp.zoom_OUT(event.xdata, event.ydata)
                ax.set_xlim(i1, i2)
                ax.set_ylim(j1, j2)
                draw()
            elif event.key is None and event.button == 'down':
                #print "Zoom in"
                i1, i2, j1, j2 = disp.zoom_IN(event.xdata, event.ydata)
                ax.set_xlim(i1, i2)
                ax.set_ylim(j1, j2)
                draw()

        change = False

        if event.button == 'up' and event.key == 'control':
            a += delta_axis
            slider_a.set_val(a)
            draw()

        elif event.button == 'down' and event.key == 'control':
            a -= delta_axis
            slider_a.set_val(a)
            draw()

        elif event.button == 'up' and event.key == 'shift':
            b += delta_axis
            slider_b.set_val(b)
            draw()

        elif event.button == 'down' and event.key == 'shift':
            b -= delta_axis
            slider_b.set_val(b)
            draw()

        elif event.button == 'up' and event.key == 'alt':
            pa += delta_angle
            slider_pa.set_val(pa)
            draw()

        elif event.button == 'down' and event.key == 'alt':
            pa -= delta_angle
            slider_pa.set_val(pa)
            draw()

        elif event.button == 'up' and (event.key == 'alt+control' or event.key == 'ctrl+alt'):
            pa += 0.5
            slider_pa.set_val(pa)
            draw()

        elif event.button == 'down' and (event.key == 'alt+control' or event.key == 'ctrl+alt'):
            pa -= 0.5
            slider_pa.set_val(pa)
            draw()

        else:
            change = False

        if change:
            ellipse.update(np.float(xcenter), np.float(ycenter),
                           np.float(a), np.float(b), np.float(pa))

    fig.canvas.mpl_connect('scroll_event', scroll_event)
    #######################

    def press_key(event):
        global a, b, pa, xcenter, ycenter, delta_axis, delta_angle, center_change
        global x_new, y_new, r1, r2, plotFits, imgplot, r1_org, r2_org, middle, mid_org, inv
        #print 'you pressed', event.key

        if event.key == 'up':
            ycenter += delta_position
            change = True
        elif event.key == 'down':
            ycenter -= delta_position
            change = True
        elif event.key == 'right':
            xcenter += delta_position
            change = True
        elif event.key == 'left':
            xcenter -= delta_position
            change = True

        elif event.key == 'ctrl+z':
            undo(event)
            change = False
        elif event.key == 'ctrl+y':
            redo(event)
            change = False

        else:
            change = False

        if change:
            ellipse.update(np.float(xcenter), np.float(ycenter),
                           np.float(a), np.float(b), np.float(pa))

        if event.key == 'q' or event.key == 'escape':
            center_change = False
            user_center.set_xdata([])
            user_center.set_ydata([])
            draw()

        if event.key == 'c' and plotFits == True:
            try:
                i1 = disp.x1
                i2 = disp.x2
                rold = r1
                k = 0.5+(event.xdata-i1)/(i2-i1)
                r1 = r1*k
                if r1 < middle:
                    r1 = (rold-middle)/2.+middle
                r2 = middle-0.05*(r1-middle)
                imgplot.set_clim(vmin=min(inv*r1, inv*r2),
                                 vmax=max(inv*r2, inv*r1))
                draw()
            except:
                pass

        if event.key == 'r' and plotFits == True:
            try:
                inv = -1
                imgplot.set_clim(vmin=min(inv*r1_org, inv*r2_org),
                                 vmax=max(inv*r2_org, inv*r1_org))
                draw()
            except:
                pass

        if center_change == True and event.key == 'enter':
            xcenter = x_new
            ycenter = y_new
            ellipse.update(np.float(xcenter), np.float(ycenter),
                           np.float(a), np.float(b), np.float(pa))
            center_change = False
            user_center.set_xdata([])
            user_center.set_ydata([])
            draw()

    def on_click(event):
        global a, b, pa, xcenter, ycenter, delta_axis, delta_angle, center_change, x_new, y_new
        #print 'you pressed', event.key, event.button, event.xdata, event.ydata, event.key

        if event.key == 'control' or event.key == 'shift':
            addOne_axis(event)
        elif event.key == 'alt':
            addOne_angle(event)

        if event.dblclick or (event.key == 'z' and event.button == 1):
            x_new = event.xdata
            y_new = event.ydata
            center_change = True
            user_center.set_xdata([x_new])
            user_center.set_ydata([y_new])
            draw()

        if center_change == True and event.button == 2:
            xcenter = x_new
            ycenter = y_new
            ellipse.update(np.float(xcenter), np.float(ycenter),
                           np.float(a), np.float(b), np.float(pa))
            center_change = False
            user_center.set_xdata([])
            user_center.set_ydata([])
            draw()

        if event.inaxes == ax:
            if event.key is None and event.button == 2:
                #print "Pan"
                i1, i2, j1, j2 = disp.pan(event.xdata, event.ydata)
                ax.set_xlim(i1, i2)
                ax.set_ylim(j1, j2)
                draw()

    fig.canvas.mpl_connect('button_press_event', on_click)

    fig.canvas.mpl_connect('key_press_event', press_key)

    annotate("Coordinates ...", (0.016, 0.73),
             xycoords='figure fraction', size=14, color='maroon')

    Ux = annotate(" ", (0.016, 0.69), xycoords='figure fraction', size=12)
    Uy = annotate(" ", (0.016, 0.65), xycoords='figure fraction', size=12)

    annotate("WCS ...", (0.016, 0.60),
             xycoords='figure fraction', size=14, color='maroon')

    Ura = annotate(" ", (0.016, 0.56), xycoords='figure fraction', size=11)
    Ualf = annotate(" ", (0.016, 0.54), xycoords='figure fraction', size=10)
    Udec = annotate(" ", (0.016, 0.50), xycoords='figure fraction', size=11)
    Udelt = annotate(" ", (0.016, 0.47), xycoords='figure fraction', size=10)

    def in_motion(event):
        #print 'you pressed', event.key, event.button, event.xdata, event.ydata, event.key, event.inaxes
        x = event.xdata
        y = event.ydata
        if event.inaxes == ax:

            Ux.set_text("X: "+'{:.2f}'.format(x))
            Uy.set_text("Y: "+'{:.2f}'.format(y))

            if HaveFitsFile:
                try:
                    world = w.wcs_pix2world([[x, y]], 1)
                    RA = world[0][0]
                    DEC = world[0][1]
                    Ura.set_text("RA: "+'{:.4f}'.format(RA))
                    Udec.set_text("DEC: "+'{:.4f}'.format(DEC))
                    c = SkyCoord(ra=RA, dec=DEC, unit=(u.degree, u.degree))
                    wcs_hmsdms = c.to_string('hmsdms', precision=4, sep=':')
                    wcs_hmsdms = wcs_hmsdms.split(" ")
                    Ualf.set_text(r"$\alpha: $"+wcs_hmsdms[0])
                    Udelt.set_text(r"$\delta: $"+wcs_hmsdms[1])
                except:
                    pass

            draw()
        else:
            Ux.set_text(" ")
            Uy.set_text(" ")
            Ura.set_text(" ")
            Udec.set_text(" ")
            Ualf.set_text(" ")
            Udelt.set_text(" ")
            draw()

    fig.canvas.mpl_connect('motion_notify_event', in_motion)

    plt.show()


#################################################################
if __name__ == '__main__':

    if os.path.isfile('foo.reg'):
        xcmd('rm foo.reg')

    if (len(sys.argv) < 2):
        print("\n Use \""+sys.argv[0]+" -h\" for help ...\n", file=sys.stderr)
        exit(1)
    opts, args = arg_parser()
    file_root = opts.object

    main(file_root)
