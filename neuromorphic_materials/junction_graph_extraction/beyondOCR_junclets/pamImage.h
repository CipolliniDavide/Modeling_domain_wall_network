/*
 * PamImage
 * By Axel Brink and Tijn van der Zant
 * (C) 2007, University of Groningen
 * Read and write binary .pbm, .pgm and .ppm.
 */

#ifndef PAMIMAGE_H
#define PAMIMAGE_H

#include <string>

struct RGBPixel
{
    unsigned char r;
    unsigned char g;
    unsigned char b;
    unsigned char m; // mask value (fills up the word size in memory)
};

typedef unsigned char GrayPixel;

typedef int IntPixel;

enum {NO_IMAGE = 0, BW_IMAGE = 1, GRAY_IMAGE = 2, RGB_IMAGE = 3, INT_IMAGE = 4};

class PamImage
{
    private:
        RGBPixel **ppRGBPixels;   // Row pointers to pixel data
        GrayPixel **ppGrayPixels; // Row pointers to pixel data
        IntPixel **ppIntPixels;   // Row pointers to pixel data
        std::string sFileName;
        int iImageType;           // NO_IMAGE = 0; BW_IMAGE = 1; GRAY_IMAGE = 2; RGB_IMAGE = 3; INT_IMAGE = 4
        void _createGrayPixels(long width, long height);
        void _createRGBPixels(long width, long height);
        void _createIntPixels(long width, long height);
        void _loadBWPixels(FILE *pFile);
        void _loadGrayPixels(FILE *pFile);
        void _loadRGBPixels(FILE *pFile);
        void _saveBW(FILE *pFile);
        void _saveGray(FILE *pFile);
        void _saveRGB(FILE *pFile);
        void _saveInt(FILE *pFile);
        long _readNumber(FILE *pFile);
        PamImage* _convertBWToGray();
        PamImage* _convertBWToRGB();
        PamImage* _convertIntToGray();
        PamImage* _convertGrayToRGB();
        PamImage* _convertGrayToBW();
        PamImage* _convertRGBToGray();
        long width;
        long height;
        bool isValid();
        void copy(PamImage const &other);

    public:
        PamImage();
        PamImage(PamImage const &other);
        PamImage &operator=(PamImage const &other); // cannot be wrapped for Python!
        PamImage(std::string _sFileName);
        PamImage(FILE *pFile);
        PamImage(int _iImageType, long lWidth, long lHeight);
        ~PamImage();
        
        void clear();
        void loadImage(FILE *pFile);
        void loadImage(std::string _sFileName);
        void save(std::string _sFileName);

        PamImage* convert(int _dstImageType);

        std::string getFileName();
        int getWidth();
        int getHeight();
        std::string getFormat();
        GrayPixel **getGrayPixels();
        RGBPixel **getRGBPixels();
        IntPixel **getIntPixels();
        int get_minval();
        int get_maxval();
        int getImageType(); //NO_IMAGE = 0, BW_IMAGE = 1, GRAY_IMAGE = 2, RGB_IMAGE = 3, INT_IMAGE = 4
        GrayPixel getPixelGray(int x, int y);
        RGBPixel getPixelRGB(float x, float y);     // uses linear interpolation
        IntPixel getPixelInt(int x, int y);
        
        void printAsciiArt();
        void putPixel(long x, long y, int grayval); // Grayscale/BW
        
};

#endif //PAMIMAGE_H
