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
 * */

#ifndef _DPFLIB_UTILS_H_
#define _DPFLIB_UTILS_H_

#include <vector>
#include <cmath>

#include "pamImage.h"

typedef struct{
	int x;
	int y;
	float value;
}dflPixel;

typedef std::vector<std::vector<dflPixel> > COCOCOS;
typedef std::vector<std::vector<float> > Table2D;

std::vector<dflPixel> dflBhmLine(dflPixel p1,dflPixel p2);

bool checkboundary(const int x,const int y, const int w,const int h);

PamImage* covert_ppm(PamImage* im);

#endif
