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

#include "dflJuncletslib.h"
#include "nms.h"
#include "dflPenWidth.h"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <cstdio>

#define BLACK 0
#define WHITE 255
#define GRAY 128


Junclets::Junclets(int _verbose, int _leg_length,int _max_penwidth, int _featdims)
{
	leg_length = _leg_length;
	max_penwidth = _max_penwidth;
	featdims = _featdims;
	verbose = _verbose;
	
	if(verbose > 0)
		silence = false;
	else
		silence = true;
}

Junclets::~Junclets()
{
}

std::vector<dflJunclets> Junclets::getJunclets(PamImage* binary)
{
	/** use the simple skeleton detection method*/
	PamImage* imSkeleton = HilditchSkeleton(binary);
	std::vector<dflPixel> candidates = candidatePoints(imSkeleton);
	delete imSkeleton;
	
	if(verbose > 0) std::cout<<"Get "<<candidates.size()<<" candidate points!\n";
	
	const int w = binary->getWidth();
	const int h = binary->getHeight();
	GrayPixel** binpt = binary->getGrayPixels();
	
	std::vector<dflJunclets> juncletlist;
	
	if(verbose > 0) std::cout<<"Start to get junclets features!\n";
	
	for(int n = 0; n < candidates.size(); ++n)
	{
		dflJunclets junclets = getJuncletsPoints(binpt,w,h,candidates[n],0.0);
		juncletlist.push_back(junclets);
	}
	
	if(verbose > 0) std::cout<<"Feature done!\n";
	
	return juncletlist;
}

void Junclets::print_feature(std::vector<dflJunclets>& featlist, std::string outfile,std::string label)
{
	FILE* fp = fopen(outfile.c_str(),"w");
	
	for(int n = 0; n < featlist.size(); ++n)
	{
		if(label.size() > 0)
			fprintf(fp,"%s ",label.c_str());
		for(int d = 0; d < featlist[n].featvec.size(); ++d)
			fprintf(fp,"%.6f ",featlist[n].featvec[d]);
		fprintf(fp,"\n");
	}
	
	fclose(fp);
}

void Junclets::print_points(std::vector<dflJunclets>& featlist, std::string outfile, std::string label)
{
	FILE* fp = fopen(outfile.c_str(),"w");
	
	for(int n = 0; n < featlist.size(); ++n)
	{
		if(label.size() > 0)
			{
			fprintf(fp,"%s ",label.c_str());
			}
		
		fprintf(fp,"%d ",featlist[n].p.x);
		fprintf(fp,"%d ",featlist[n].p.y);
		fprintf(fp,"\n");
	}
	
	fclose(fp);
}


/** @dominant_angle is the baseline direction in order to achieve rotation invariant */
dflJunclets Junclets::getJuncletsPoints(GrayPixel** binpt,const int width, const int height,dflPixel here, float dominant_angle)
{
	int binsize = featdims;
	float binstep = 2*M_PI / (float) binsize;
	
	dflPixel widthend;
	
	std::vector<float> featvec;

	std::vector<dflPixel> endlist;
	
	dflPenWidth obj_penwidth;
	
	for(int i = 0; i < binsize; ++i)
	{
		float perp_ang = fmodf(dominant_angle +  binstep * i, 2*M_PI);
					
		dflPixel end;
		end.x = here.x + (int) round((float) (max_penwidth) * cos(perp_ang));
		end.y = here.y + (int) round((float) (max_penwidth) * sin(perp_ang)); 
		
		float w = obj_penwidth.measure_penwidth(binpt,width,height,here,end,max_penwidth,widthend); 
		
		if(widthend.x < 0 || widthend.y < 0)
			widthend = end;
	
		endlist.push_back(widthend);	
		featvec.push_back(w);
	}

	

	double sum = 0;
	
	for(int i = 0; i < featvec.size(); ++i){
		sum += featvec[i];
	}
	if(sum == 0) sum = 1;
	for(int i = 0; i < featvec.size(); ++i)
		featvec[i] /= sum;
	
	
	dflJunclets kp;
	kp.p = here;
	kp.dominant_angle = dominant_angle;
	kp.featvec = featvec;
	kp.endlist = endlist;
	return kp;	
}

std::vector<dflPixel> Junclets::candidatePoints(PamImage* imSkeleton)
{
	std::vector<dflPixel> forkpoints = mpsSearchForkpoints(imSkeleton);
	std::vector<dflStroke> substrokes = mpsSearchSubStroke(imSkeleton);
	std::vector<dflPixel> highcurvaturepoints = dfl2JunctionCandidate(substrokes);
	std::vector<dflPixel> candidates;
	for(int n = 0; n < forkpoints.size(); ++n)
		candidates.push_back(forkpoints[n]);
	for(int n = 0; n < highcurvaturepoints.size(); ++n)
		candidates.push_back(highcurvaturepoints[n]);
	
	return candidates;
}


std::vector<dflPixel> Junclets::dfl2JunctionCandidate(std::vector<dflStroke>& stroke)
{
	std::vector<dflPixel> candi;

	for(int n = 0; n < stroke.size(); ++n)
	{
		std::vector<dflPixel> skelist = stroke[n].skeleton;
		
		std::vector<float> angle;
		for(int t = 0; t < skelist.size(); ++t)
		{
			int cy = skelist[t].y;
			int cx = skelist[t].x;
			
			dflPixel p1,p2;
			int idp1 = t - leg_length;
			if(idp1 < 0) idp1 = 0;
			p1 = skelist[idp1];
			
			int idp2 = t + leg_length;
			if(idp2 > skelist.size()-1 ) idp2 = skelist.size()-1;
			p2 = skelist[idp2];
			
			float ang1 = (float)atan2((double)(cy-p1.y),(double)(cx-p1.x));
			if(ang1 < 0.0) ang1 += 2*M_PI;
			
			float ang2 = (float)atan2((double)(cy-p2.y),(double)(cx-p2.x));
			if(ang2 < 0.0) ang2 += 2*M_PI;
			
			float tang_ang = std::fabs((double)(ang1-ang2));
			tang_ang = M_PI - std::min((double)tang_ang,2*M_PI-tang_ang);
			angle.push_back(tang_ang);
		}
		
		std::vector<float> peak;
		nms1d_cir(angle,angle.size(),10,peak);
		
		for(int t = 1; t < peak.size()-1; ++t)
		{
			if(peak[t] > 0)
			 candi.push_back(skelist[t]);
		}
	}
	return candi;
}

std::vector<dflStroke> Junclets::mpsSearchSubStroke(PamImage* imSkeleton)
{
	const int w = imSkeleton->getWidth();
	const int h = imSkeleton->getHeight();
	GrayPixel** skept = imSkeleton->getGrayPixels();
	
	bool* used = new bool[w*h];
	for(int i = 0; i < w*h; ++i)
		used[i] = true;
	
	std::vector<dflStroke> strokelist;
	
	int nCnt;
	for(int y = 0; y < h; ++y)
		for(int x = 0; x < w; ++x)
	{
		if(skept[y][x] == 0 && used[y*w+x] )
		{
			std::vector<dflPixel> tmplist;
			dflPixel p; p.x = x; p.y = y;
			tmplist.push_back(p);
			used[y*w+x] = false;

			std::vector<dflPixel> endpoints;
			
			for(int nt = 0; nt < tmplist.size(); ++nt)
			{
				int cy = tmplist[nt].y;
				int cx = tmplist[nt].x;
				
				nCnt = 0;
				
				for(int sy = -1; sy <= 1; ++sy)
					for(int sx = -1; sx <= 1; ++sx)
				{
					if(checkboundary(cx+sx,cy+sy,w,h) && skept[cy+sy][cx+sx]==BLACK)
					{
						nCnt++;
						if(used[(cy+sy)*w+cx+sx])
						{
							dflPixel p; p.x = cx+sx; p.y = cy+sy;
							tmplist.push_back(p);
							used[(cy+sy)*w+cx+sx] = false;
						}
					}
				}
				
				if(nCnt == 2)
				{
					dflPixel p; p.x = cx; p.y = cy;
					endpoints.push_back(p);
				}
				
				if(nCnt > 3)
				{
					if(!silence) std::cout<<"There are still forp points exist!\n";
				}
			} // end of search
			
			int size = tmplist.size();
			int idx = rand() % size + 2;
			
			if(idx > size-1) idx = size - 1;
			if(idx < 0) idx = 0;
			if( endpoints.size() == 0)
			{
				endpoints.push_back(tmplist[idx]);
				endpoints.push_back(tmplist[(idx+1)%size - 1]);
			}
			
			dflStroke sk;
			sk.endlist = endpoints;
			strokelist.push_back(sk);
			
		}
	}
	
	/** make order of the sample */
	for(int i = 0; i < w*h; ++i)
		used[i] = true;
	
	for(int n = 0; n < strokelist.size(); ++n)
	{
		if(strokelist[n].endlist.size() != 2)
			if(!silence)std::cout<<"the end points "<<(int)strokelist[n].endlist.size()<<" should be 2!\n";
		
		dflPixel end = strokelist[n].endlist[0];
			
		std::vector<dflPixel> search;
		search.push_back(end);
		used[end.y*w+end.x] = false;
			
		for(int nt = 0; nt < search.size(); ++nt)
		{
			int cy = search[nt].y;
			int cx = search[nt].x;

			bool wbreak = false;

			 for(int sy = -1; sy <= 1; ++sy){
				for(int sx = -1; sx <= 1; ++sx)
				{
					if(checkboundary(cx+sx,cy+sy,w,h) && skept[cy+sy][cx+sx]==0 && used[(cy+sy)*w+cx+sx])
					{
						dflPixel p; p.x = cx+sx; p.y = cy+sy;
						search.push_back(p);
						used[(cy+sy)*w+cx+sx] = false;
						wbreak = true; 
						break;	
					}
				}
				if(wbreak) break;
			}
			
		}//end of search
		
		strokelist[n].skeleton = search;	
	}
	
	delete[]used;
	return strokelist;
}

/** find all fork points 
 * and remove these points on the imSkeleton image */
std::vector<dflPixel> Junclets::mpsSearchForkpoints(PamImage* imSkeleton)
{
	const int w = imSkeleton->getWidth();
	const int h = imSkeleton->getHeight();
	GrayPixel** skept = imSkeleton->getGrayPixels();
	
	std::vector<dflPixel> forklist;
	
	int nCnt;
	for(int y = 0; y < h; ++y)
		for(int x = 0; x < w; ++x)
	{
		if(skept[y][x] == BLACK)
		{
			nCnt = 0;
			for(int sy = -1; sy <= 1; ++sy)
				for(int sx = -1; sx <= 1; ++sx)
			{
				if(checkboundary(x+sx,y+sy,w,h) && skept[y+sy][x+sx]==0)
					nCnt++;
			}
			
			if(nCnt > 3)
			{
				dflPixel p;
				p.x = x; p.y = y;
				forklist.push_back(p);
				skept[y][x] = WHITE; // turn the black point to white
			}
		}
	}	
	return forklist;
}

/** the following code is for Skeleton line detection */
int Junclets::func_nc8(int* b)
{
	int n_odd[4] = {1,3,5,7};
	int j,sum,d[10];
	
	for(int i = 0; i <= 9; ++i)
	{
		j = i;
		if(j == 9) j = 1;
		if(std::abs(*(b+j)) == 1)
			d[i] = 1;
		else
			d[i] = 0;
	}
	
	sum = 0;
	for(int i = 0; i < 4; ++i)
	{
		j = n_odd[i];
		sum = sum + d[j] -d[j]*d[j+1]*d[j+2];
	}
	return sum;
}


PamImage* Junclets::HilditchSkeleton(PamImage* bin)
{

	int wid = bin->getWidth();
	int hei = bin->getHeight();
	GrayPixel** binpt = bin->getGrayPixels();
	
	/*for(int y = 0; y < hei; ++y)
		for(int x = 0; x < wid; ++x)
			binpt[y][x] = 255 - binpt[y][x];*/
	
	/** hilditch method for thinning.
	 * 	the code is from: http://cis.k.hosei.ac.jp/~wakahara/Hilditch.c */
	 
	//bin->save("bin.ppm");
	 
	int offset[9][2] = {{0,0},{1,0},{1,-1},{0,-1},{-1,-1},
		      {-1,0},{-1,1},{0,1},{1,1} }; /* offsets for neighbors */
	int n_odd[4] = { 1, 3, 5, 7 };      /* odd-number neighbors */
	int px, py;                         /* X/Y coordinates  */
	int b[9];                           /* gray levels for 9 neighbors */
	int condition[6];                   /* valid for conditions 1-6 */
	int counter;                        /* number of changing points  */
	int i, x, y, copy, sum;             /* control variable          */
	
	PamImage* res = new PamImage(2,wid,hei);
	GrayPixel** respt = res->getGrayPixels();
	for(int iy = 0; iy < hei; ++iy)
		for(int ix = 0; ix < wid; ++ix)
			respt[iy][ix] = binpt[iy][ix];
			
	
	do{
		//std::cout<<"Start to "<<iter<<"-th iteration!\n";
		
		counter = 0;
		for(int iy = 0; iy < hei; ++iy)
			for(int ix = 0; ix < wid; ++ix)
			{
				for(int is = 0; is < 9; ++is)
				{
					b[is] = 0;
					px = ix + offset[is][0];
					py = iy + offset[is][1];
					if(px >=0 && px < wid && py >=0 && py < hei)
					{
						if(respt[py][px] == BLACK)
							b[is] = 1;
						else if(respt[py][px] == GRAY)
							b[is] = -1;
							
					}
				}
				
				for(int is = 0; is < 6; ++is)
					condition[is] = 0;
				
				if(b[0] == 1)
					condition[0] = 1;
				
				sum = 0;
				for(int is = 0; is < 4; ++is)
					sum = sum + 1 -std::abs(b[n_odd[is]]);
				if(sum >= 1)
						condition[1] = 1;
				
				sum = 0;
				for(int is = 1; is <=8; ++is)
					sum = sum + std::abs(b[is]);
				
				if(sum >= 2) 
					condition[2] = 1;
				
				sum = 0;
				for(int is = 1; is <= 8; ++is)
					if(b[is]==1) sum++;
				
				if( sum >= 1)
					condition[3] = 1;
				
				if(func_nc8(b) == 1)
					condition[4] = 1;
				
				sum = 0;
				for(int is = 1; is <= 8; is++)
				{
					if(b[is] != -1)
						sum++;
					else
					{
						copy = b[is];
						b[is] = 0;
						if(func_nc8(b) == 1)
							sum++;
						b[is] = copy;
					}
				}
				
				if(sum == 8)
					condition[5] = 1;
				
				if(condition[0] && condition[1] && condition[2] &&
				   condition[3] && condition[4] && condition[5])
				   {
					   respt[iy][ix] = GRAY;
					   counter++;
				   }
			}
			
		if(counter != 0)
		{
			for(int iy = 0; iy < hei; ++iy)
				for(int ix = 0; ix <wid; ++ix)
					if(respt[iy][ix] == GRAY)
						respt[iy][ix] = WHITE;
		}	
	}while(counter != 0);
	
	/* remove the 4-connect to 8 -connect */
	/* [ 0 1 1
	 *   0 1 0
	 *   0 1 0] */
	
	 int nn ;
	for(int y = 0; y < hei; ++y)
		for(int x = 0; x < wid; ++x)
	{
		if(respt[y][x] == BLACK)
		{
			nn = 0;			
			for(int iy = -1; iy <= 1; ++iy)
				for(int ix = -1; ix <= 1; ++ix)
			{
				if(y+iy >=0 && y+iy <hei && x+ix>=0 && x+ix<wid)
				{
					if(respt[y+iy][x+ix]==BLACK)
						nn++;
				}
			}
			
			if(nn > 2)
			{
				if(y-1 >=0)
				{
					if(respt[y-1][x]==BLACK)
					{
						/*if(x-1>=0 && respt[y-1][x-1]==BLACK){
							respt[(y-1)][x] = WHITE;
							if(y-2 >=0 && x+1 < wid && respt[y-2][x+1] == BLACK)
								respt[y-1][x+1] = BLACK;
						}
							
								
						if(x+1<wid && respt[(y-1)][x+1]==BLACK){
							respt[(y-1)][x] = WHITE;
							if(y-2 >=0 && x-1 >= 0 && respt[y-2][x-1] == BLACK)
								respt[y-1][x-1] = BLACK;
						}*/
						
						if(x-1>=0 && respt[y-1][x-1]==BLACK){
							//respt[(y-1)][x] = WHITE;
							if(y-2 >=0 && x+1 < wid && respt[y-2][x+1] != BLACK)
								respt[(y-1)][x] = WHITE;
						}
							
								
						if(x+1<wid && respt[(y-1)][x+1]==BLACK){
							//respt[(y-1)][x] = WHITE;
							if(y-2 >=0 && x-1 >= 0 && respt[y-2][x-1] != BLACK)
								respt[(y-1)][x] = WHITE;
						}
								
					}
				}
				
				if(y+1 < hei)
				{
					if(respt[(y+1)][x]==BLACK)
					{
						/*if(x-1>=0 && respt[(y+1)][x-1]==BLACK){
							respt[(y+1)][x] = WHITE;
							if(y+2 < hei && x+1 < wid && respt[y+2][x+1] == BLACK)
								respt[y+1][x+1] = BLACK;
						}
						
						if(x+1<wid && respt[(y+1)][x+1]==BLACK){
							respt[(y+1)][x] = WHITE;
							if(y+2 < hei && x-1 >=0 && respt[y+2][x-1] == BLACK)
								respt[y+1][x-1] = BLACK;
						}*/
						if(x-1>=0 && respt[(y+1)][x-1]==BLACK){
							//respt[(y+1)][x] = WHITE;
							if(y+2 < hei && x+1 < wid && respt[y+2][x+1] != BLACK)
								respt[(y+1)][x] = WHITE;
						}
						
						if(x+1<wid && respt[(y+1)][x+1]==BLACK){
							//respt[(y+1)][x] = WHITE;
							if(y+2 < hei && x-1 >=0 && respt[y+2][x-1] != BLACK)
								respt[(y+1)][x] = WHITE;
						}
					}
				}
				
				if(x-1 >=0)
				{
					if(respt[y][x-1]==BLACK)
					{
						/*if(y-1>=0 && respt[(y-1)][x-1]==BLACK){
							respt[y][x-1] = WHITE;
							if(x-2 >= 0 && y+1 < hei && respt[y+1][x-2] == BLACK)
								respt[y+1][x-1] = BLACK;
						}
								
						if(y+1<hei && respt[(y+1)][x-1]==BLACK){
							respt[y][x-1] = WHITE;
							if(x-2 >= 0 && y-1 >=0 && respt[y-1][x-2] == BLACK)
								respt[y-1][x-1] = BLACK;						
						}*/
						if(y-1>=0 && respt[(y-1)][x-1]==BLACK){
							//respt[y][x-1] = WHITE;
							if(x-2 >= 0 && y+1 < hei && respt[y+1][x-2] != BLACK)
								respt[y][x-1] = WHITE;
						}
								
						if(y+1<hei && respt[(y+1)][x-1]==BLACK){
							//respt[y][x-1] = WHITE;
							if(x-2 >= 0 && y-1 >=0 && respt[y-1][x-2] != BLACK)
								respt[y][x-1] = WHITE;
						}
								
					}
				}
				
				if(x+1 < wid)
				{
					if(respt[y][x+1]==BLACK)
					{
						/*if(y-1>=0 && respt[(y-1)][x+1]==BLACK){
							respt[y][x+1] = WHITE;
							if(x+2 < wid && y+1 < hei && respt[y+1][x+2] == BLACK)
								respt[y+1][x+1] = BLACK;
						}	
								
						if(y+1<hei && respt[(y+1)][x+1]==BLACK){
							respt[y][x+1] = WHITE;
							if(x+2 < wid && y-1 >=0 && respt[y-1][x+2] == BLACK)
								respt[y-1][x+1] = BLACK;
						}*/
						if(y-1>=0 && respt[(y-1)][x+1]==BLACK){
							//respt[y][x+1] = WHITE;
							if(x+2 < wid && y+1 < hei && respt[y+1][x+2] != BLACK)
								respt[y][x+1] = WHITE;
						}	
								
						if(y+1<hei && respt[(y+1)][x+1]==BLACK){
							//respt[y][x+1] = WHITE;
							if(x+2 < wid && y-1 >=0 && respt[y-1][x+2] != BLACK)
								respt[y][x+1] = WHITE;
						}
								
					}
				}
				
			}
		}
	}
	
	return res;
}
