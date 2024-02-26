#include "pamImage.h"
#include "dflBinarylib.h"

#include <iostream>
#include <cmath>
#include <vector>

typedef struct{
	int y;
	int x;
}edgePos;

dplBinaryMethod::dplBinaryMethod(PamImage im,bool removeLargeArea):_im(im)
{
    _width = _im.getWidth();
    _height = _im.getHeight();
  
    _windx = 40;
    _windy = 40;
    
    _min_area = 0.01 * std::max(_width,_height);
    
    _removeLargeArea = removeLargeArea;
    
    _bin = new PamImage(2,_width,_height);
}

dplBinaryMethod::~dplBinaryMethod()
{
    //_bin->save("bin.ppm");
}
/*************** main run *************************/
PamImage* dplBinaryMethod::run(int para)
{
    if(para == OTSU)
       otsu_binary(); 
       
    if(para == NIBLACK)
    {
        double k = 0.5;        
        setWindows();             
        niblack_binary(k);
    }
    
    if(para==SAUVOLA)
    {
        double k = 0.2;        
        setWindows();             
        sauvola_binary(k);
    }
    
    if(para==GATOS)
    {
        PamImage* g = wienerFilter(3);
        double k = 0.2;
        setWindows();
        GrayPixel** imp = _im.getGrayPixels();
        GrayPixel** gp  = g->getGrayPixels();
        for(int y=0;y<_height;++y)
            for(int x=0;x<_width;++x)
                imp[y][x] = gp[y][x];
               
        sauvola_binary(k);
        //_bin->save("sauvola.ppm"); 
        PamImage* b=backgroundEstimation(_bin,g,65);
        gatos_threshold(b);
        
        b->save("bk.ppm");
    }
    
    removeSmallRegions();
    

    return _bin;
}

void dplBinaryMethod::removeSmallRegions(int min_area)
{
	if(min_area < 0)
		min_area = _min_area;
	
	const int w = _width;
	const int h = _height;
	
	GrayPixel** binpt = _bin->getGrayPixels();
	
	
	bool* used = new bool[w*h];
	for(int i = 0; i < w*h; ++i)
		used[i] = false;
	
	std::vector<std::vector<edgePos> > cocos;
	
	for(int y = 0; y < h; ++y)
		for(int x = 0; x < w; ++x)
	{
		
		if(used[y*w+x] == false && binpt[y][x] == 0)
		{
			std::vector<edgePos> plist;
			
			edgePos p; 
			p.y = y; p.x = x;
			plist.push_back(p);
			used[y*w+x] = true;
			
			for(int nt = 0; nt < (int)plist.size(); ++nt)
			{
				int cy = plist[nt].y;
				int cx = plist[nt].x;
				
				for(int sy = -1; sy <= 1; ++sy)
					for(int sx = -1; sx <= 1; ++sx)
				{
					if( sx != 0 || sy != 0)
						if(cy+sy>=0 && cy+sy < h && cx+sx >= 0 && cx+sx <w && used[(cy+sy)*w+cx+sx] == false)
						{
							if( binpt[cy+sy][cx+sx] == 0 )
							{
								edgePos p;
								p.y = cy+sy; p.x = cx+sx;
								plist.push_back(p);
								used[(cy+sy)*w+cx+sx] = true;
							}
						}
				}
			}
			
			cocos.push_back(plist);
			
			if((int)plist.size() < min_area)
			{
				/* remove this region */
				for(int i = 0; i < (int)plist.size(); ++i)
					binpt[plist[i].y][plist[i].x] = 255;
			}
		}
	}
	
	if(_removeLargeArea)
	{
		int aver = 0;
		float sigma = 0;
		int nCnt = cocos.size();
		for(int i = 0; i < nCnt; ++i)
			aver += cocos[i].size();
		
		aver /= nCnt;
		for(int i = 0; i < nCnt; ++i)
			sigma = (cocos[i].size()-aver)*(cocos[i].size()-aver);
		
		sigma = sqrt(sigma/(double)nCnt);
		
		//std::cout<<"get aver: "<<aver<<" sigma:"<<sigma<<"\n";
		
		for(int i = 0; i < nCnt; ++i)
		{
			//std::cout<<cocos[i].size()<<" "<<aver+3*sigma<<" \n";
			
			if(cocos[i].size() > (aver + 9 * sigma * sigma))
			{
				for(int t = 0; t < (int)cocos[i].size(); ++t)
					binpt[cocos[i][t].y][cocos[i][t].x] = 255; 
			}
		}
	}
	
	
	
	delete[]used;
}
 
void dplBinaryMethod::setWindows(int wx,int wy)
{
    if( wx==0 || wy == 0)
    {
        if(_windy == 0 )
           _windy = (int) ( 2.0 * _height ) /3;
        if(_windx == 0)
            _windx = _width-1 < _windy ? _width-1 : _windy;
    }
    else
    {
        _windy = wy;
        _windx = wx;
    }
}
void dplBinaryMethod::thresholdFixed(size_t th)
{
    GrayPixel** imp = _bin->getGrayPixels();
    GrayPixel** oim = _im.getGrayPixels();
    
    for(int y = 0; y < _height; ++y)
        for(int x = 0; x < _width; ++x)
        {
            if( oim[y][x] < th )
                imp[y][x] = 0;
            else
                imp[y][x] = 255;
        }
}

void dplBinaryMethod::thresholdSurface(float** surface)
{
    GrayPixel** oim = _im.getGrayPixels();
    GrayPixel** imp = _bin->getGrayPixels();
    
    for(int y=0;y<_height;++y)
        for(int x=0;x<_width;++x)
        {
            if(oim[y][x] > surface[y][x])
                imp[y][x] = 255;
            else
                imp[y][x] = 0;
        }
}

size_t dplBinaryMethod::otsu_threshold()
{
    float *histogram = new float[256];
    float *omega     = new float[256];
    float *mu        = new float[256];
    
    for(size_t i=0; i < 256; ++i)
    {
        histogram[i] = 0;
        omega[i]     = 0;
        mu[i]        = 0;
    }
    
    /* get histogram of graypixels */
    GrayPixel** pim = _im.getGrayPixels();
    for(int y = 0; y < _height; ++y)
        for(int x = 0; x < _width; ++x)
            ++histogram[ (int)pim[y][x] ];
            
    float total = _width * _height;
    for(size_t i = 0 ; i < 256; ++i )
        histogram[i] /= total;
    
    /* calculate omega */
    omega[0] = histogram[0];
    for(size_t i = 1; i < 256; ++i)
        omega[i] = omega[i-1] + histogram[i];
    
    /* calculate mu. Note: mu[0] = 0 */
    for(size_t i = 1; i < 256; ++i)
        mu[i] = mu[i-1] + i*histogram[i];
    
    /* determine otsu threshold */
    float maxvalue = 0;
    size_t threshold = 0;
    for(size_t i = 0; i < 256; ++i)
    {
        if(omega[i] * (1-omega[i]) > 0 )
        {
            float value =  mu[255] * omega[i];
            value -= mu[i];
            value  = pow(value,2);
            value /= ( omega[i] * (1 - omega[i]) );
            if(value > maxvalue)
            {
                maxvalue = value;
                threshold = i;
            }
        }
    } 
    delete [] histogram;
    delete [] omega;
    delete [] mu;
    
    return threshold;
}
void dplBinaryMethod::otsu_binary()
{
    //std::cout<<"starting: otsu binarization method!\n";
    size_t threshold = otsu_threshold();
    //std::cout<<"get threshold: "<<threshold<<std::endl;
    thresholdFixed(threshold);
}

void dplBinaryMethod::niblack_binary(double k)
{
    //std::cout<<"starting: niblack binarization method!\n";
    //std::cout<<"window size is:("<<_windx<<" "<<_windy<<")\n";
    
    float** surface;
    surface = malloc_2D(_width,_height,0.0);
    getSurfaceNiblack(surface,k);
    //std::cout<<"Get surface down!\n";
    thresholdSurface(surface);
    free_2D(surface,_width,_height);
}

void dplBinaryMethod::sauvola_binary(double k)
{
    //std::cout<<"starting: saulva binarization method!\n";
    //std::cout<<"window size is:("<<_windx<<" "<<_windy<<")\n";
    
    float** surface;
    surface = malloc_2D(_width,_height,0.0);
    getSurfaceSauvola(surface,k);
    thresholdSurface(surface);
   // std::cout<<"end of sauvola binary !\n";
    
    free_2D(surface,_width,_height);
}
double dplBinaryMethod::localStatsWindow(float** mat_mu,float** mat_var)
{
    if(mat_mu==NULL || mat_var==NULL){
        std::cout<<"Error: the space for mu and variance should be malloced!\n";
        throw 1;
    }
        
    int wx = _windx / 2;
    int wy = _windy / 2;
    
    GrayPixel** imp = _im.getGrayPixels();
    
    double max_var = 0;
    double sum_mu,sum_var;
    
    double mu,var;
    double windarea = _windx * _windy;

    double value;

    /* sliding window */

        for(int y = wy; y < _height-wy; ++y)
        {
            sum_mu = 0;
            sum_var = 0;
            //for(int ssy = 0; ssy < wy; ++ssy){
              for(int sy=0;sy<_windy;++sy){
                for(int sx = 0; sx < _windx; ++sx)
                {
                    value = imp[y-wy+sy][sx];
                    sum_mu  += value;
                    sum_var += value*value;
                    
                }
            }    

            mu  = sum_mu  / windarea;
            var = sqrt( (sum_var - (sum_mu*sum_mu)/windarea)/windarea );
     
            if( var > max_var)
                max_var = var;
            mat_mu [y][wx] = mu;
            mat_var[y][wx] = var;
     
            for(int x = 1; x < _width-_windx; ++x)
            {
                /* remove the left value and add the right value */

                for(int sy = 0; sy < _windy; ++sy)
                {
                    value = imp[y-wy+sy][x-1];
                    sum_mu  -= value;
                    sum_var -= value*value;
                    
                    value = imp[y-wy+sy][x+_windx-1];
                    sum_mu  += value;
                    sum_var += value*value;
                }
                
                mu = sum_mu / windarea;
                var = sqrt( (sum_var - (sum_mu*sum_mu)/windarea)/windarea );
                
                if(var<0)
                    std::cout<<"err:\n";
                    
                if( var > max_var)
                    max_var = var;
                
                mat_mu [y][x+wx] = mu;
                mat_var[y][x+wx] = var;
            } 
        }
     
    return max_var;
}
void dplBinaryMethod::getSurfaceNiblack(float** mat_surface,double k)
{
    //double max_var;
    float** mat_mu,**mat_var;
    
    mat_mu  = malloc_2D(_width,_height,0.0);
    mat_var = malloc_2D(_width,_height,0.0);
    
        
    //std::cout<<"debug:start window!\n";
    //double max_var = localStatsWindow(mat_mu,mat_var);
    //print_2D(mat_mu,_width,_height);
    //std::cout<<"debug:end window!\n";
    //print_2D(mat_var,_width,_height);
    int wx = _windx / 2;
    int wy = _windy / 2;
    
    double mu,var;

    
    for(int y = wy; y < _height-wy; ++y)
        for(int x = wx; x < _width-wx;++x)
        {
            mu  = mat_mu [y][x];
            var = mat_var[y][x];
            mat_surface[y][x] = mu + k*var;
            //std::cout<<mu<<" ";
        }
     
 
    /* filled top-left corner */
    for(int y=0;y<wy;++y)
        for(int x=0;x<wx;++x)
            mat_surface[y][x] = mat_surface[wy][wx];
    /* top line */
    for(int y = 0; y< wy; ++y)
        for(int x=wx;x<_width-wx;++x)
            mat_surface[y][x] = mat_surface[wy][x];
    /* top-right corner */
    for(int y=0;y<wy;++y)
        for(int x = _width-wx;x<_width;++x)
            mat_surface[y][x] = mat_surface[wy][_width-wx-1];
    
    
    for(int y=_height-wy;y<_height;++y)
    {
        /* down-left corner */
        for(int x=0;x<wx;++x)
            mat_surface[y][x] = mat_surface[_height-wy-1][wx];
        /* down line */
        for(int x = wx; x< _width-wx;++x)
            mat_surface[y][x] = mat_surface[_height-wy-1][x];
        /* down-right corner */
        for(int x = _width-wx; x < _width; ++x)
            mat_surface[y][x] = mat_surface[_height-wy-1][_width-wx-1];
        
    }
    
    /* left line */
    for(int y=wy;y<_height-wy;++y)
        for(int x = 0;x<wx;++x)
            mat_surface[y][x] = mat_surface[y][wx];
    /* right line */
    for(int y=wy;y<_height-wy;++y)
        for(int x = _width-wx;x<_width;++x)
            mat_surface[y][x] = mat_surface[y][_width-wx-1];
            
    free_2D(mat_mu,_width,_height);
    free_2D(mat_var,_width,_height);
}

void dplBinaryMethod::getSurfaceSauvola(float** mat_surface,double k)
{
    //double max_var;
    float** mat_mu,**mat_var;
    double dR = 128;
    
    mat_mu  = malloc_2D(_width,_height,0.0);
    mat_var = malloc_2D(_width,_height,0.0);
    
        

    //double max_var = localStatsWindow(mat_mu,mat_var);

    int wx = _windx / 2;
    int wy = _windy / 2;
    
    double mu,var;

    
    for(int y = wy; y < _height-wy; ++y)
        for(int x = wx; x < _width-wx;++x)
        {
            mu  = mat_mu [y][x];
            var = mat_var[y][x];
            mat_surface[y][x] = mu*(1+k*(var/dR - 1));
            //std::cout<<mu<<" ";
        }
     
 
    /* filled top-left corner */
    for(int y=0;y<wy;++y)
        for(int x=0;x<wx;++x)
            mat_surface[y][x] = mat_surface[wy][wx];
    /* top line */
    for(int y = 0; y< wy; ++y)
        for(int x=wx;x<_width-wx;++x)
            mat_surface[y][x] = mat_surface[wy][x];
    /* top-right corner */
    for(int y=0;y<wy;++y)
        for(int x = _width-wx;x<_width;++x)
            mat_surface[y][x] = mat_surface[wy][_width-wx-1];
    
    
    for(int y=_height-wy;y<_height;++y)
    {
        /* down-left corner */
        for(int x=0;x<wx;++x)
            mat_surface[y][x] = mat_surface[_height-wy-1][wx];
        /* down line */
        for(int x = wx; x< _width-wx;++x)
            mat_surface[y][x] = mat_surface[_height-wy-1][x];
        /* down-right corner */
        for(int x = _width-wx; x < _width; ++x)
            mat_surface[y][x] = mat_surface[_height-wy-1][_width-wx-1];
        
    }
    
    /* left line */
    for(int y=wy;y<_height-wy;++y)
        for(int x = 0;x<wx;++x)
            mat_surface[y][x] = mat_surface[y][wx];
    /* right line */
    for(int y=wy;y<_height-wy;++y)
        for(int x = _width-wx;x<_width;++x)
            mat_surface[y][x] = mat_surface[y][_width-wx-1];
            
    free_2D(mat_mu,_width,_height);
    free_2D(mat_var,_width,_height);
}

float** dplBinaryMethod::malloc_2D(int w,int h,float v)
{
    float**mat;
    mat = new float*[h];
    for(int pt = 0; pt < h; ++pt)
        mat[pt] = new float[w];
    
    for(int py = 0; py < h; ++py)
        for(int px = 0; px < w; ++px)
            mat[py][px] = v;
    return mat;
}

void dplBinaryMethod::free_2D(float** mat,int w,int h)
{
    for(int pt = 0 ; pt < h; ++pt)
        delete [] mat[pt];
    delete [] mat;
    mat = NULL;
}
void dplBinaryMethod::print_2D(float** mat,int w,int h)
{
    for(int y=0;y<h;++y)
    {
        for(int x=0;x<w;++x)
            std::cout<<mat[y][x]<<" ";
        std::cout<<std::endl;
    }
}

PamImage* dplBinaryMethod::wienerFilter(int windows)
{
    float** mat_mu;
    float** mat_var;
    
    mat_mu  = malloc_2D(_width,_height,0.0);
    mat_var = malloc_2D(_width,_height,0.0);
    
    double mu,var;
    int win = windows/2;
    int winsize = windows*windows;
    
    GrayPixel** imp = _im.getGrayPixels();
    int px,py;
    
    for(int y = 0; y < _height; ++ y)
    {
        
        for(int x = 0; x < _width; ++x)
        {
            mu = 0;
            var = 0;
            
            for(int suby = -win; suby <=win;++suby)
                for(int subx = -win; subx <=win;++subx)
                {
                    py = y+suby;
                    px = x+subx;
                    if( py < 0) py = 0;
                    if( py > _height-1) py =  _height -1;
                    if( px < 0) px = 0;
                    if( px > _width -1 ) px = _width - 1;
                    
                    mu  += imp[py][px];
                    var += imp[py][px]*imp[py][px];
                }
           
           mat_mu [y][x] = mu / winsize;
           mat_var[y][x] = ( var - (mu*mu)/winsize )/ winsize;
        }
    }
    
    PamImage * wienerIm = new PamImage(2,_width,_height);
    GrayPixel** wim = wienerIm->getGrayPixels();
    
    double value,ratio;
    
    for(int y = 0; y < _height; ++y)
        for(int x = 0; x < _width; ++x)
        {
            var = 0;
            for(int suby = -win; suby <=win;++suby)
                for(int subx = -win; subx <=win;++subx)
                {
                    py = y+suby;
                    px = x+subx;
                    if( py < 0) py = 0;
                    if( py > _height-1) py =  _height -1;
                    if( px < 0) px = 0;
                    if( px > _width -1 ) px = _width - 1;
                    
                    var += mat_var[py][px];
                }
            var /= winsize;
            
            if(mat_var[y][x]>0)
            {
                ratio = var / mat_var[y][x];
                if(ratio > 1) ratio = 1;
                if(ratio < 0 ) ratio = 0;
                value = (float)imp[y][x] - ratio * (float)imp[y][x] + ratio * mat_mu[y][x];
            }
            else
                value = (float) imp[y][x];
            
           /* if(mat_var[y][x]>0)
                value  = mat_mu[y][x] + ( (mat_var[y][x] - var)*((float)imp[y][x] -mat_mu[y][x]) ) / mat_var[y][x];
            else 
                value = mat_mu[y][x];*/
                
            if(value > 255) value = 255;
            if(value < 0) value = 0;
            wim[y][x] = (unsigned char) value;
        }
    free_2D(mat_mu,  _width,_height);
    free_2D(mat_var, _width,_height);
    return wienerIm;
}

PamImage* dplBinaryMethod::backgroundEstimation(PamImage* mask,PamImage* wieIm,int windows)
{
    PamImage* bk = new PamImage(2,_width,_height);
    int win = windows / 2;
    
    GrayPixel** mkp = mask->getGrayPixels();
    GrayPixel** imp = wieIm->getGrayPixels();
    GrayPixel** bkp = bk->getGrayPixels();
    
    double sum,sumsize;
    int px,py;
    /*
    for(int y = 0; y < _height; ++ y)
    {
        sum = 0;
        sumsize = 0;
        for(int suby = -win; suby<=win;++suby)
            for(int subx = -win; subx<=win;++subx)
            {
                px = 0+subx;
                py = y+suby;
                if(px<0) px = 0;
                if(px>_width-1)px=_width-1;
                if(py<0) py = 0;
                if(py>_height-1)py=_height-1;
                
                sum     += imp[py][px] *  (mkp[py][px]/255);
                sumsize += (  mkp[py][px] ) / 255;
            }
        
        if(mkp[y][0]==255)
            bkp[y][0] = imp[y][0];
        else
            bkp[y][0] = (unsigned char)(sum / sumsize);
        
        for(int x = 1; x < _width; ++x)
        {
            for(int suby = -win; suby<=win; ++suby)
            {
                px = x-win-1;
                py = y+suby;
                if(px<0) px = 0;
                if(px>_width-1)px=_width-1;
                if(py<0) py = 0;
                if(py>_height-1)py=_height-1;
                
                sum     -= imp[py][px] *  (mkp[py][px]/255);
                sumsize -= (  mkp[py][px] ) / 255;
                
                px = x + win;
                if(px<0) px = 0;
                if(px>_width-1)px=_width-1;
                
                sum     += imp[py][px] *  (mkp[py][px]/255);
                sumsize += (  mkp[py][px] ) / 255;
            }
            
            if(mkp[y][x]==255)
                bkp[y][x] = imp[y][x];
            else
                bkp[y][x] = (unsigned char)(sum / sumsize );
            
        }
    }
    */
    
    for(int y = 0; y < _height; ++y)
        for(int x = 0; x < _width; ++x)
        {
            sum = 0; 
            sumsize = 0;
            
            if(mkp[y][x] == 255 )
                bkp[y][x] = imp[y][x];
            else
            {
                for(int suby = -win; suby <= win; ++suby)
                    for(int subx = -win; subx <= win; ++subx)
                    {
                        px = x + subx;
                        py = y + suby;
                        if(px > 0 && px < _width-1 && py > 0 && py < _height -1)
                        {
                            sum += imp[py][px] *(mkp[py][px]/255);
                            sumsize += mkp[py][px]/255;
                            //std::cout<<sum<<" "<<sumsize<<" "<<mkp[py][px]/255<<std::endl;
                        }
                    }
                if(sumsize > 0)
                    bkp[y][x] = (unsigned char)( sum / sumsize);
                else
                    bkp[y][x] = 0;
            }
        } 
    return bk;    
}

void dplBinaryMethod::gatos_threshold(PamImage* bk)
{
    //float** surface;
    GrayPixel** bkp = bk->getGrayPixels();
    GrayPixel** imp = _im.getGrayPixels();
    GrayPixel** bip = _bin->getGrayPixels();
    double delta;
    double sum = 0; 
    double sumsize = 0;
    
    for(int y = 0; y < _height; ++y)
        for(int x = 0; x < _width; ++x)
        {
            sum +=  bkp[y][x] - imp[y][x];
            sumsize += (255 - bip[y][x] )/255;
        }
    delta = sum / sumsize;
    
    //std::cout<<"estimate delta: "<<delta<<std::endl;
    double b;
    sum = 0;sumsize = 0;
    
    for(int y=0;y<_height;++y)
        for(int x=0;x<_width;++x)
        {
            sum += bkp[y][x] * ( bip[y][x] / 255 );
            sumsize += bip[y][x] / 255;
        }
    b = sum / sumsize;
    
    double q = 0.6, p1=0.5,p2=0.8;
    double value,th;
    
    //surface = malloc_2D(_width,_height,0.0);    
    for(int y = 0; y < _height; ++y)
        for(int x = 0; x < _width; ++x)
        {
            value = (1-p1)/(1+exp( (-4*bkp[y][x])/(b*(1-p1)) + (2*(1+p1))/(1-p1))) + p2;
            th = q*delta*value;
            if( bkp[y][x]-imp[y][x] > th )
                bip[y][x] = 0;
            else
                bip[y][x] = 255;
        }
    //free_2D(surface,_width,_height);
}
