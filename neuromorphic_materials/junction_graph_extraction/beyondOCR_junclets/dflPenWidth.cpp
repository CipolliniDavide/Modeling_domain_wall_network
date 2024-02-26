/* This source file is a part of 
 * 
 * Document feature library (DFLib)
 * 
 * @copyright: Sheng He, University of Groningen (RUG)
 * 
 * Email: heshengxgd@gmail.com
 * 
 *  27, Nov., 2015
 * 
 * 
 *    This is the source code for Junclets feature in the paper:
 * 
 *    Writer identification using directional ink-trace width measurements
 *        A.A. Brinka, J. Smitb,  M.L. Bulacua, L.R.B. Schomakera, 
 *    Pattern Recognition, (45) 2012 pp.162--171
 * 
 * */
 
#include "dflPenWidth.h"
#include <cmath>

dflPenWidth::dflPenWidth()
{
}
dflPenWidth::~dflPenWidth()
{
}

inline void dflPenWidth::swap_int(int &val1, int &val2)
{
    int temp = val1;
    val1 = val2;
    val2 = temp;
}

// Return euclidean distance
inline float dflPenWidth::euclid_dist(int x1, int y1, int x2, int y2)
{
    int dx = x2 - x1;
    int dy = y2 - y1;
    return std::sqrt((float) (dx * dx + dy * dy));
}

// Return pixel intensity if withing boundaries, else return -1
inline int dflPenWidth::get_pixel(GrayPixel** pixels, int x, int y, int width, int height)
{
    if ((x >= 0) && (x < width) && (y >= 0) && (y < height))
    {
        return pixels[y][x];
    }
    else
    {
        return -1;
    }
}

int dflPenWidth::abs(int a)
{
	if(a<0) return -a;
	return a;
}

float dflPenWidth::measure_penwidth(GrayPixel** pixels,
                              long width, long height,
                              dflPixel from, dflPixel to,
                              int max_penwidth,dflPixel& endp) {
    bool in_ink = false;
    
    bool steep = (abs(to.y - from.y) > abs(to.x - from.x));
    if (steep) {
        swap_int(from.x, from.y);
        swap_int(to.x, to.y);
    }

    int deltax = abs(to.x - from.x);
    int deltay = abs(to.y - from.y);
    int error = -deltax / 2;
    int xstep; // new
    int ystep;
    int y = from.y;
    int old_y;
    if (from.y < to.y) {
        ystep = 1;
    }
    else {
        ystep = -1;
    }
    if (from.x < to.x) {
        xstep = 1;
    }
    else {
        xstep = -1;
    }
    int old_x;
    int x = from.x;
    int* real_x_p;
    int* real_y_p;
    if (steep) {
        real_x_p = &y;
        real_y_p = &x;
    } else {
        real_x_p = &x;
        real_y_p = &y;
    }
    endp.y = -1; endp.x = -1;
    dflPixel ink_begin;
    ink_begin.x = -1; ink_begin.y = -1; // initialization just to make the compiler happy
    bool done = false;
    old_x = x;
    old_y = y;
    while (not done) {
        int pixelval; 
        //if (steep) {
            //pixelval = get_pixel(pixels, y, x, width, height);
        //} else {
            //pixelval = get_pixel(pixels, x, y, width, height);
        //}
        pixelval = get_pixel(pixels, *real_x_p, *real_y_p, width, height); // slower?
        if (pixelval == 0) {
            if (!in_ink) {
                ink_begin.x = x;
                ink_begin.y = y;
                in_ink = true;
            }
        } else {                      // light pixel encountered
            if (in_ink) {
                // we've run out of ink pixels; we can stop.
                endp.y = old_y; endp.x = old_x;
                if(steep) swap_int(endp.y,endp.x);
                return euclid_dist(x, y, ink_begin.x, ink_begin.y);

            }
        }

        error += deltay;
        if (error > 0) {
			old_y = y;
            y += ystep;
            error -= deltax;
        }
        if (x == to.x) done = true;
        old_x = x;
        x += xstep;
        //std::cout<<y<<" "<<x<<"\n";
    }
    if (!in_ink) return 0.0;    // no ink found (error)
    else return (float) (max_penwidth + 1); // end of ink not found
}
