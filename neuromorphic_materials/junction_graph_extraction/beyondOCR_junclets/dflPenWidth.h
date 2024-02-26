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
#ifndef _DFL_PENWIDTH_H_
#define _DFL_PENWIDTH_H_

#include "pamImage.h"
#include "dflUtils.h"

class dflPenWidth{
	public:
		float measure_penwidth(GrayPixel** pixels,
                              long width, long height,
                              dflPixel from, dflPixel to,
                              int max_penwidth,dflPixel& endp);
        dflPenWidth();
        ~dflPenWidth();
	private:
		inline int get_pixel(GrayPixel** pixels, int x, int y, int width, int height);
		inline float euclid_dist(int x1, int y1, int x2, int y2);
		inline void swap_int(int &val1, int &val2);
		int abs(int a);
	
};


#endif
