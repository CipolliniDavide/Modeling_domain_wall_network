/* This source file is a part of 
 * 
 * Document feature library (DFLib)
 * 
 * @copyright: Sheng He, University of Groningen (RUG)
 * 
 * Email: heshengxgd@gmail.com
 * 
 *  26, Nov., 2015
 * 
 * */
 
#ifndef _DFL_BINARYLIB_H_
#define _DFL_BINARYLIB_H_

#include "pamImage.h"


/** the result binary foreground is 0, and background is 255 */
enum{OTSU=0,NIBLACK,SAUVOLA,GATOS};

class dplBinaryMethod{
    public:
        dplBinaryMethod(PamImage im,bool removeLargeArea);
        ~dplBinaryMethod();
        
        /** the main function: *
         * @param: para = 0  -> otsu
         *         para = 1  -> niblack
         * 		   para = 2  -> sauvola
         *         para = 3  -> gatos */
         
        PamImage* run(int para);
        void setWindows(int wx=0,int wy=0);

    private:
    
        PamImage _im;
        PamImage* _bin;
        
        /* otsu method */
        size_t otsu_threshold();
        void otsu_binary();
        
        /* niblack method */
        double localStatsWindow(float** mat_mu,float** mat_var);
        void  getSurfaceNiblack(float** mat_surface,double k);
        void niblack_binary(double k);
        
        void removeSmallRegions(int min_area = -1);
        
        /* Sauvola method */
        void getSurfaceSauvola(float** mat_surface,double k);
        void sauvola_binary(double k);
        
        void thresholdFixed(size_t th);
        void thresholdSurface(float** surface);
        
        
        /* gatos method */
        PamImage* wienerFilter(int windows);
        PamImage* backgroundEstimation(PamImage* mask,PamImage* wieIm,int windows);
        void gatos_threshold(PamImage* bk);
        
        float** malloc_2D(int w,int h,float v);
        void free_2D(float**mat,int w,int h);
        void print_2D(float** mat,int w,int h);
        int _width;
        int _height;
        int _windx;
        int _windy;
        
        int _min_area;
        bool _removeLargeArea;
        
};

#endif
