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

#include "dflUtils.h"


std::vector<dflPixel> dflBhmLine(dflPixel p1,dflPixel p2)
{
	
	int x1 = p1.x, x2 = p2.x;
	int y1 = p1.y, y2 = p2.y;
	
	
	std::vector<dflPixel> plist;
	plist.push_back(p1);
	
	int x,y,dx,dy,dx1,dy1,px,py,xe,ye,i;
	dx=x2-x1;
	dy=y2-y1;
	dx1=std::fabs(dx);
	dy1=std::fabs(dy);
	px=2*dy1-dx1;
	py=2*dx1-dy1;
	
	if(dy1<=dx1)
	{
		if(dx>=0)
		{
			x=x1;
			y=y1;
			xe=x2;
		}
		else
		{
			x=x2;
			y=y2;
			xe=x1;
		}	
		dflPixel p;
		p.x = x;
		p.y = y;
		plist.push_back(p);
		for(i=0;x<xe;i++)
		{
			x=x+1;
			if(px<0)
			{
				px=px+2*dy1;
			}
			else
			{
				if((dx<0 && dy<0) || (dx>0 && dy>0))
				{
					y=y+1;
				}
				else
				{
					y=y-1;
				}
				px=px+2*(dy1-dx1);
			}
			
			dflPixel p;
			p.x = x;
			p.y = y;
			plist.push_back(p);
		}
	}
	else
	{
		if(dy>=0)
		{
			x=x1;
			y=y1;
			ye=y2;
		}
		else
		{
			x=x2;
			y=y2;
			ye=y1;
		}
		
		dflPixel p;
		p.x = x;
		p.y = y;
		plist.push_back(p);
		
		for(i=0;y<ye;i++)
		{
			y=y+1;
			if(py<=0)
			{
				py=py+2*dx1;
			}
			else
			{
				if((dx<0 && dy<0) || (dx>0 && dy>0))
				{
					x=x+1;
				}
				else
				{
				 x=x-1;
				}
				py=py+2*(dx1-dy1);
			}
			
			dflPixel p;
			p.x = x;
			p.y = y;
			plist.push_back(p);
		}
	}
	
	plist.push_back(p2);
	return plist;
}	


bool checkboundary(const int x,const int y, const int w,const int h)
{
	if(x >= 0 && x < w && y >=0 && y < h)
		return true;
	return false;
}

PamImage* covert_ppm(PamImage* im)
{
	const int w = im->getWidth();
	const int h = im->getHeight();
	GrayPixel** impt=im->getGrayPixels();
	
	PamImage* imc = new PamImage(2,w,h);
	GrayPixel** imcpt = imc->getGrayPixels();
	
	for(int y = 0; y < h; ++y)
		for(int x = 0; x < w; ++x){
			//std::cout<<(int)impt[y][x]<<" ";
			if(impt[y][x]==0)
				imcpt[y][x] = 0;
			else
				imcpt[y][x] = 255;
		}
	return imc;
}
