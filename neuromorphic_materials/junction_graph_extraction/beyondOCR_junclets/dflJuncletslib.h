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
 *    Junction detection in handwritten documents and its application to writer identification
 *    Sheng He,Marco Wiering, Lambert Schomaker
 *    Pattern Recognition, (48) 2015 pp.4036--4048
 * 
 * */

#ifndef _DFL_JUNCLETDLIB_H_
#define _DFL_JUNCLETDLIB_H_

#include <vector>
#include <string>

#include "dflUtils.h"
#include "pamImage.h"

typedef struct{
	dflPixel p;
	float dominant_angle;
	std::vector<float> featvec;
	std::vector<dflPixel> endlist;
}dflJunclets;

typedef struct{
	std::vector<dflPixel> skeleton;
	std::vector<dflPixel> endlist;
}dflStroke;

class Junclets{
	public:
		Junclets(int _verbose=0, int _leg_length=100,int _max_penwidth=200, int _featdims=120);
		std::vector<dflJunclets> getJunclets(PamImage* binary);
		
		void print_feature(std::vector<dflJunclets>& featlist, std::string outfile,std::string label);
		void print_points(std::vector<dflJunclets>& featlist, std::string outfile,std::string label);
		
		~Junclets();
	
	private:

		dflJunclets getJuncletsPoints(GrayPixel** binpt,const int width, const int height,dflPixel here, float dominant_angle);
		/** candidate points computation */
		std::vector<dflPixel> candidatePoints(PamImage* imSkeleton);
		std::vector<dflPixel> mpsSearchForkpoints(PamImage* imSkeleton);
		std::vector<dflStroke> mpsSearchSubStroke(PamImage* imSkeleton);
		std::vector<dflPixel> dfl2JunctionCandidate(std::vector<dflStroke>& stroke);
		
		/** skeleton */
		PamImage* HilditchSkeleton(PamImage* bin);
		int func_nc8(int* b);
		
		
		int leg_length;
		int max_penwidth;
		int featdims;             // dimension of feature
		
		int verbose;
		bool silence;
};


/** an example to cal Junclets */

/*
#include "pamImage.h"
#include "dflBinarylib.h"
#include "dflUtils.h"
#include "dflJuncletslib.h"

#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

int main(int argc,char*argv[])
{
	if(argc!=3)
	{
		std::cout<<"Usage: "<<argv[0]<<" ppm outfile\n";
		return 1;
	}
	
	PamImage* im = new PamImage(argv[1]);
	
	if(im->getImageType()!=2)
		im=im->convert(2);
	
	dplBinaryMethod bin(*im,false);
	PamImage* otsu=bin.run(0);


	
	const int w = im->getWidth();
	const int h = im->getHeight();
	
	Junclets junc;
	
	std::vector<dflJunclets> feature = junc.getJunclets(otsu);
	
	std::string label;
	junc.print_feature(feature,std::string(argv[2]),label);
	
	std::cout<<"feat size: "<<feature.size()<<"\n";

	
	delete im;
	delete otsu;
	return 1;
}*/


#endif
