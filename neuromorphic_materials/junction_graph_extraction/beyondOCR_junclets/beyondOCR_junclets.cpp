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

//#define DEBUG 1

typedef struct{
	dflPixel point;
	int number;
}linevec;

std::vector<float> getCOLDLogPolar(std::vector<linevec>& veclist,int nrad=7,int nang=12,int silence=1,float scale=5.0)
{
	double lpbase = pow(10,log10((double)nrad)/nrad);
	double angbase = 2 * M_PI / (double) nang;
			
	std::vector<double> radiiQuants(nrad);	

	std::vector<std::vector<float> > featvec(nrad);
	for(int n = 0; n < nrad; ++n)
		featvec[n].resize(nang,0.0);
	
	int nloss = 0;
		
	for (int i=0; i<nrad; ++i){
		radiiQuants[i] = (pow(lpbase,i+1)-1)/(nrad - 1);
	}
	
	double s_radius = (double)scale / radiiQuants[0];

	for(int i = 0; i < nrad; ++i){	
		radiiQuants[i] = radiiQuants[i] * s_radius + scale;
	}

#ifdef DEBUG
	if(silence) std::cout<<"Starting to debug!\n";
	
	for(int i = 0; i < nrad; ++i)	
		std::cout<<i+1<<"-th radius: "<<radiiQuants[i]/radiiQuants[nrad-1]*10.5<<"\n";
		
	int center = radiiQuants[nrad-1]+0.5;
	const int w = 2*center+1;
	const int h = 2*center+1;
	PamImage* im = new PamImage(3,w,h);
	RGBPixel** impt = im->getRGBPixels();
	for(int my = 0; my < h; ++my)
		for(int mx = 0; mx < w; ++mx){
			impt[my][mx].r=impt[my][mx].b=impt[my][mx].g=255;
			//impt[my][mx].r=0;
			//impt[my][mx].g=0;
			//impt[my][mx].b=255;
		}
	
	double max_num = 0;
	for(int nv = 0; nv < veclist.size(); ++nv)
	{
		float e_radius = sqrt(veclist[nv].point.y*veclist[nv].point.y+veclist[nv].point.x*veclist[nv].point.x);
		
		if(e_radius > radiiQuants[nrad-1]) {
			continue;
		}
		else
		{
			 if(veclist[nv].number>max_num)
				max_num = veclist[nv].number;
		}
	}
	
	for(int nv = 0; nv < veclist.size(); ++nv)
	{
		float e_radius = sqrt(veclist[nv].point.y*veclist[nv].point.y+veclist[nv].point.x*veclist[nv].point.x);
		
		if(e_radius > radiiQuants[nrad-1]) {
			continue;
		}
		else
		{
			int color = (int)((double)veclist[nv].number/max_num*255+60);
			if(color>255)color=255;
			
			impt[veclist[nv].point.y+center][veclist[nv].point.x+center].r = color;
			impt[veclist[nv].point.y+center][veclist[nv].point.x+center].g = 0;//255-(int)((double)veclist[nv].number/max_num*255);
			impt[veclist[nv].point.y+center][veclist[nv].point.x+center].b = 255-color;//255-(int)((double)veclist[nv].number/max_num*255);
		}
	}
	
	/*for(int my = 0; my < h; ++my)
		for(int mx = 0; mx < w; ++mx)
	{
		int cy = my - center;
		int cx = mx - center;
		float dis = sqrt(cy*cy+cx*cx);
		bool found = false;
		for(int nr = 0; nr < nrad; ++nr)
		{
			if(std::fabs(dis-radiiQuants[nr])<0.5)
				found = true;
		}
		if(found) {
			impt[my][mx].r = 0;
			impt[my][mx].g = 255;
			impt[my][mx].b = 0;
		}
	}*/
	
	im->save("debug_cold.ppm");
	delete im;
	
#endif	
	for(int nv = 0; nv < (int)veclist.size(); ++nv)
	{
		float ang = atan2(veclist[nv].point.y,veclist[nv].point.x);
		if(ang < 0.0) ang += M_PI*2;
		
		float e_radius = sqrt(veclist[nv].point.y*veclist[nv].point.y+veclist[nv].point.x*veclist[nv].point.x);
		
		if(e_radius > radiiQuants[nrad-1]) {
			nloss++;
			continue;
		}
		
		int angbin = ang / angbase;
		if(angbin >= nang) angbin--;
		if(angbin < 0) angbin = 0;
		
		int radbin = -1;
		for(int r = 0; r < nrad; ++r)
		{
			if(e_radius < radiiQuants[r])
			{
				radbin = r;
				break;
			}
		}
		
		if(radbin >= 0 && radbin < nrad)
			featvec[radbin][angbin] += veclist[nv].number;
	}
	
	if(silence)std::cout<<"loss points: "<<(float)nloss/(float)veclist.size()*100<<"\%\n";
	
	std::vector<float> feature;
	double sum = 0.0;
	for(int nr = 0; nr < nrad; ++nr)
		for(int nv = 0; nv < nang; ++nv)
			sum += featvec[nr][nv];
	if(sum==0) sum = 1;
	for(int nr = 0; nr < nrad; ++nr){
		for(int nv = 0; nv < nang; ++nv){
			feature.push_back(featvec[nr][nv]/sum);
#ifdef DEBUG
			std::cout<<nv<<"/"<<featvec[nr][nv]/sum*1000<<", ";
#endif
		}
#ifdef DEBUG
		std::cout<<"\n";
#endif
	}
	return feature;
}

std::vector<linevec> getLineVector(std::vector<dflJunclets>& junclets)
{
	std::vector<linevec> veclist;
	
	int dx,dy;
	
	
	for(int np = 0; np < (int)junclets.size(); ++np)
	{
		
		
		for(int d = 0; d < (int)junclets[np].endlist.size(); ++d)
		{
			dx = junclets[np].endlist[d].x-junclets[np].p.x;
			dy = junclets[np].endlist[d].y-junclets[np].p.y;
		
			bool found = false;
			int findx = -1;
			for(int sv = 0; sv < (int)veclist.size(); ++sv)
			{
				if(veclist[sv].point.y == dy && veclist[sv].point.x == dx)
				{
					found = true;
					findx = sv;
					break;
				}
			}
			
			if(found)
				veclist[findx].number++;
			else
			{
				linevec lv;
				lv.point.x = dx;
				lv.point.y = dy;
				lv.number = 1;
				veclist.push_back(lv);
			}
		}	
	}
	return veclist;
}

int main(int argc,char*argv[])
{
	if(argc!=6)
	{
		std::cout<<"Usage: "<<argv[0]<<" ppm label outfile model binary(IAM-0&Firemaker1)\n";
		return 1;
	}
	
	PamImage* im = new PamImage(argv[1]);
	//std::string label(argv[2]);
	std::string label;
	std::string outfile(argv[3]);
	int model=atoi(argv[4]);
	int binary = atof(argv[5]);

	
	if(im->getImageType()!=2)
		im=im->convert(2);
	
	PamImage* otsu;
	
	if(binary > 0)
	{
		dplBinaryMethod bin(*im,false);
		otsu=bin.run(0);
		std::string save_binary= "_binary.ppm";
		std::string fullName_binary = outfile.c_str() + save_binary;
		otsu->save(fullName_binary);
	}
	else
		otsu = covert_ppm(im);

	const int w = im->getWidth();
	const int h = im->getHeight();
	
	Junclets junc;
	
	std::vector<dflJunclets> feature = junc.getJunclets(otsu);
	
	std::cout<<"get junclets: "<<feature.size()<<"\n";
	
	std::string feat_name= "_features.txt";
	std::string point_name= "_points.txt";
	std::string fullName_feat = outfile.c_str() + feat_name;
	std::string fullName_points = outfile.c_str() + point_name;
	
	FILE* fp = fopen(outfile.c_str(),"w");

	for(int n = 0; n < feature.size(); ++n)
	{
		fprintf(fp,"%d %d ",n,(int)feature[n].featvec.size());
		for(int t = 0; t < feature[n].featvec.size(); ++t)
			fprintf(fp,"%.6f ",feature[n].featvec[t]);
		fprintf(fp,"\n");
	}

	fclose(fp);
	
	junc.print_feature(feature,fullName_feat.c_str(),label);
	junc.print_points(feature,fullName_points.c_str(),label);

    //std::vector<linevec> veclist;
    //std::vector<linevec> veclist= getLineVector(junc);
	//getCOLDLogPolar(veclist);

	delete im;
	delete otsu;
	return 1;
}
