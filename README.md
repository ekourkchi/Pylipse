## Table of contents
1. [Pylipse](#Pylipse)
2. [How to run](#run)
3. [Options](#Options)
4. [Help](#Help)

## Pylipse <a name="Pylipse"></a>

A GUI for manual ellipse fitting for Elliptical and Spiral galaxies ...

![demo_pylipse](https://user-images.githubusercontent.com/13570487/74600799-c375b400-5053-11ea-90b0-a7546a07cfb0.png)


## How to run <a name="run"></a>

            python pylipse.py [options]
            
### Example
 
            python  pylipse.py -j  NGC4037 

## Options <a name="Options"></a>

            Use the `-h` flag to see all the following options.

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



            Options:
              -h, --help            show this help message and exit
              -j OBJECT, --object=OBJECT
                                    The object name

 
## Help <a name="Help"></a>

### Mouse Actions
Use mouse middle wheel for scrolling

 A) When mouse pointer is on the image

   - scroll-down  = zoom-in
   - scroll-up    = zoom-outer
   - middle-click = re-center the image

 B) When anywhere on the GUI window
 
   - Ctrl+Scroll_up   = increase the semi-major axis
   - Ctrl+Scroll_down = decrease the semi-major axis
 
   - Shift+Scroll_up   = increase the semi-minor axis
   - Shift+Scroll-down = decrease the semi-minor axis

   - Alt+Scroll_up   = increase the position angle (PA)
   - Alt+Scroll_down = decrease the position angle (PA)
 
   - PA increases in clock-wise direction. 
   - PA = 0 if semi-major axis is horizontal
   - Step-size is displayed in the box next to the a-/b- and PA-control bars.
 
   - `Left  click` on the box: increases the step size by one pixel
   - `Right click` on the box: decreases the step size by one pixel
   
   - `Left/Right double click` = choose a new center for the ellipse
   - `Middle click` or `enter-key` = re-center the ellipse, if a new center has been already chosen
   - `Middle double click` = choose the new center and re-draw the ellipse at once (b7+b8)
   -  z+Left_click = choose a new center for the ellipse (b7)
   - q/Esc = Ignoring the new chosen center 

 C) When the fits file is displayed (no jpg file)
 
   - chaning the contrast
   - reset the contrst parameters

   - - - -
   
 * Version : v1.0 (March 2016)
 * Author: *Ehsan Kourkchi* <ekourkchi@gmail.com>
 * Copyright 2016: Feel free to distribute and modify for a better performance
