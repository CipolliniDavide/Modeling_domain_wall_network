/*
 * Read and write PAM (.pbm, .pgm, .ppm, .pnm) images
 * Axel Brink, 2006
 * University of Groningen
 */

#include "pamImage.h"
#include <iostream>
#include <cmath>
#include <cstdio>

PamImage::PamImage()
{
    width = 0;
    height = 0;
    sFileName = "";
    iImageType = NO_IMAGE;
}

PamImage::PamImage(PamImage const &other)
{
	copy(other);
}

PamImage &PamImage::operator=(PamImage const &other)
{
	if(this != &other) //skip the copy if we assign ourselves
	{
		clear();
		copy(other);
	}
	return *this;
}

PamImage::PamImage(std::string _sFileName)
{
    iImageType = NO_IMAGE;
    PamImage::loadImage(_sFileName);
}

// Load an image from a pointer to an open file
PamImage::PamImage(FILE *pFile)
{
    iImageType = NO_IMAGE;
    PamImage::loadImage(pFile);
}

// Create empty (black) image
PamImage::PamImage(int _iImageType, long _lWidth, long _lHeight)
{
    if (_lWidth <= 0 || _lHeight <= 0) {
        std::cerr << "Error in PamImage: cannot create image with dimensions " << _lWidth << "x" << _lHeight << std::endl;
        throw 1;
    } else if ((_iImageType == BW_IMAGE) || (_iImageType == GRAY_IMAGE))
    {
        PamImage::_createGrayPixels(_lWidth, _lHeight);
        for (long row = 0; row < _lHeight; row++)
        {
            for (long column = 0; column < _lWidth; ++column)
            {
                ppGrayPixels[row][column] = 0; // initialize as 0
            }
        }
        //std::cerr << "Debug: created " << _lWidth << "x" << _lHeight << " gray/bw pamImage." << std::endl;
    }
    else if (_iImageType == RGB_IMAGE)
    {
        PamImage::_createRGBPixels(_lWidth, _lHeight);
        for (long y = 0; y < _lHeight; y++) {
            for (long x = 0; x < _lWidth; ++x) {
               ppRGBPixels[y][x].r = 0; // initialize as 0
               ppRGBPixels[y][x].g = 0; // initialize as 0
               ppRGBPixels[y][x].b = 0; // initialize as 0
               ppRGBPixels[y][x].m = 0; // initialize as 0
           }
        }
    }
    else if (_iImageType == INT_IMAGE)
    {
        PamImage::_createIntPixels(_lWidth, _lHeight);
        for (long y = 0; y < _lHeight; y++)
        {
            for (long x = 0; x < _lWidth; ++x)
            {
                ppIntPixels[y][x] = 0; // initialize as 0
            }
        }
    }
    else
    {
        std::cerr << "Cannot create empty Pam image: illegal image type: " << _iImageType << std::endl;
        throw 1;
    }
    iImageType = _iImageType;
    width = _lWidth;
    height = _lHeight;
}

PamImage::~PamImage()
{
    //std::cerr << "Debug: destroying " << width << "x" << height << " pamImage." << std::endl;
    clear();
}


// Copies the info from other into this object
// private, for public use you have the copy constructor or assignment operator
void PamImage::copy(PamImage const &other)
{
	//copy variables
	sFileName = other.sFileName;
	height = other.height;
	width = other.width;
	iImageType = other.iImageType;
		
	switch(iImageType)
	{
		case BW_IMAGE:
		case GRAY_IMAGE:
		{
			//copy graypixel pointer
			ppGrayPixels = new GrayPixel*[height];
			for(int row = 0; row < height; ++row)
			{
				ppGrayPixels[row] = new GrayPixel[width];
				for(int column = 0; column < width; ++column)
				{
					ppGrayPixels[row][column] = other.ppGrayPixels[row][column];
				}
			}
			break;
		}
		case RGB_IMAGE:
		{
			//copy rgb pixel pointer
			ppRGBPixels = new RGBPixel*[height];
			for(int row = 0; row < height; ++row)
			{
				ppRGBPixels[row] = new RGBPixel[width];
				for(int column = 0; column < width; ++column)
				{
					ppRGBPixels[row][column] = other.ppRGBPixels[row][column];
				}
			}
			break;
		}
		case INT_IMAGE:
		{			
			//copy int pixel pointer
			ppIntPixels = new IntPixel*[height];
			for(int row = 0; row < height; ++row)
			{
				ppIntPixels[row] = new IntPixel[width];
				for(int column = 0; column < width; ++column)
				{
					ppIntPixels[row][column] = other.ppIntPixels[row][column];
				}
			}	
			break;
		}
		case NO_IMAGE:
			break;
	}
}

void PamImage::clear()
// release image; free memory
{
    if ((iImageType == BW_IMAGE) || (iImageType == GRAY_IMAGE))
    {
        int row;
        for (row = 0; row < height; row++) {
            delete[] ppGrayPixels[row];
        }
        delete[] ppGrayPixels;
    }
    else if (iImageType == RGB_IMAGE)
    {
        int row;
        for (row = 0; row < height; row++) {
            delete[] ppRGBPixels[row];
        }
        delete[] ppRGBPixels;
    }
    else if (iImageType == INT_IMAGE)
    {
        for (long y = 0; y < height; y++) {
            delete[] ppIntPixels[y];
        }
        delete[] ppIntPixels;
    }
    else if (iImageType == NO_IMAGE)
    {
        // very nice: nothing to do
    }
    else
    {
        std::cerr << "Cannot clear image; unknown type: " << iImageType << ".\n";
        throw 1;
    }
    sFileName = "";
    width = 0;
    height = 0;
    iImageType = NO_IMAGE;
}

// Create an (uninitialized) array for a gray image
void PamImage::_createGrayPixels(long w, long h)
{
    ppGrayPixels = new GrayPixel*[h];
    for (long y = 0; y < h; y++)
    {
        ppGrayPixels[y] = new GrayPixel[w];
    }   
}

// Create an (uninitialized) array for an RGB image
void PamImage::_createRGBPixels(long w, long h)
{
    ppRGBPixels = new RGBPixel*[h];
    for (long y = 0; y < h; y++) {
        ppRGBPixels[y] = new RGBPixel[w];
    }
}

// Create an (uninitialized) array for an int image
void PamImage::_createIntPixels(long w, long h)
{
    ppIntPixels = new IntPixel*[h];
    for (long y = 0; y < h; y++) {
        ppIntPixels[y] = new IntPixel[w];
    }
}

// Load pixels from black/white image file (P4). In memory: 0 = black; 1 = white.
// 8 pixels per byte; last byte in a row can contain don't-care bits.
void PamImage::_loadBWPixels(FILE *pFile)
{
    PamImage::_createGrayPixels(width, height);
    long bytesperline = (width % 8 == 0 ? width / 8 : width / 8 + 1);
    unsigned char *pixelline_in = new unsigned char[bytesperline];
    
    for (long y = 0; y < height; ++y) {
        fread(pixelline_in, 1, bytesperline, pFile);
        unsigned char *in_ptr = pixelline_in;
        GrayPixel *pixelline_out = ppGrayPixels[y];
        int bitnum = 7;                    // most significant bit first
        
        for (long x = 0; x < width; ++x)
        {
            pixelline_out[x] = ((*in_ptr & (1 << bitnum)) == 0 ? 1 : 0); // extract a bit
            bitnum -= 1;
            if (bitnum < 0)
            {
                if (x < width - 1)
                {
                    bitnum = 7;
                    in_ptr += 1;
                }
            }   
        }
    }
    delete[] pixelline_in;
}

// Load pixels from grayscale image file (P5).
void PamImage::_loadGrayPixels(FILE *pFile)
{
    PamImage::_createGrayPixels(width, height);
    for (long y = 0; y < height; y++) {
        fread(ppGrayPixels[y], 1, width, pFile);
    }
}

// Load pixels from color image file (P6).
void PamImage::_loadRGBPixels(FILE *pFile)
{
    PamImage::_createRGBPixels(width, height);
    unsigned char *pixelline = new unsigned char[3 * width];
    
    for (long y = 0; y < height; ++y) {
        fread(pixelline, 1, 3 * width, pFile);
        unsigned char *in_ptr = pixelline;
        RGBPixel *out_ptr = ppRGBPixels[y];
        for (long x = 0; x < width; ++x)
        {
            out_ptr->r = in_ptr[0];
            out_ptr->g = in_ptr[1];
            out_ptr->b = in_ptr[2];
            out_ptr->m = 0;         // or whatever
            in_ptr += 3;
            out_ptr += 1;
        }
    }
    delete[] pixelline;
}

long PamImage::_readNumber(FILE *pFile)
{
    /* Read a string upto and including the next whitespace character;
     * exclude pnm comments. Return -1 on error.
     */
    bool inItem = false;
    bool done = false;
    char c;
    int theNumber = 0;
    
    while (!done)
    {
        fscanf(pFile, "%c", &c);
        if ((c == 9) || (c == 10) || (c == 13) || (c == 32))
        {
            // whitespace
            if (inItem)
            {
                //done = true;
                return theNumber;
            }
        }
        else if (c == '#')
        {
            // comment: skip until newline
            char commentchar = '\0';
            while ((commentchar != 10) && (commentchar != 13))
            { 
                fscanf(pFile, "%c", &commentchar);
            }
        }
        else if ((c >= '0') && (c <= '9'))
        {
            theNumber = 10 * theNumber + (c - '0');
            inItem = true;
        }
        else
        {
            std::cerr << "Error: illegal character in file header" << std::endl;
            done = true;
        }
    }
    return -1;
}

// load .pbm (P4), .pgm (P5) or .ppm (P6)
void PamImage::loadImage(FILE *pFile)
{
    char magicP;  // The leading 'P' in the file
    int type;     // 4, 5, or 6; defines image type
    int gmax = 0; // Maximum gray value; not used any further.
    fscanf(pFile, "%c", &magicP);
    
    if (magicP != 'P') {
        std::cerr << "Error: not a pbm/pgm/ppm/pnm image." << std::endl;
        throw 1;
    }
    fscanf(pFile, "%i", &type);
    width = _readNumber(pFile);
    height = _readNumber(pFile);
    
    // Read gmax
    if ((type == 5) || (type == 6))
    {
        //fscanf(pFile, "%i", &gmax);
        gmax = _readNumber(pFile);
        if (gmax > 255)
        {
            std::cerr << "Error: cannot read .pgm/.ppm images that are not 8 bit per pixel." << std::endl;
            throw 1;
        }
        //fscanf(pFile, "%c", &dummy); // 1 whitespace character
    }
    
    if (type == 4) // P4
    {
        iImageType = BW_IMAGE;
        _loadBWPixels(pFile);
    }
    else if (type == 5) // P5
    {
        iImageType = GRAY_IMAGE;
        _loadGrayPixels(pFile);
    }
    else if (type == 6) // P6
    {
        iImageType = RGB_IMAGE;
        _loadRGBPixels(pFile);
    }
    else if ((type == 1) || (type == 2) || (type == 3))
    {
        std::cerr << "Error: Cannot handle ascii format P" << type << "." << std::endl;
        throw 1;
    }
    else
    {
        std::cerr << "Unknown image type: P" << type << std::endl;
        throw 1;
    }
    fclose(pFile);
}

// load .pgm or .ppm: read file and write imageInfo and ppRGBPixels/ppGrayPixels
void PamImage::loadImage(std::string _sFileName)
{
    FILE *pFile; // file pointer

    clear();
    
    PamImage::sFileName = _sFileName;
    pFile = fopen(_sFileName.c_str(), "r");
    if (pFile == NULL)
    {
       std::cerr << "Unable to open file " << _sFileName.c_str() << std::endl;
       throw 1;
    }
    else
    {
        loadImage(pFile);
    }
}

// Save .pbm file (P4) to an open file stream.
void PamImage::_saveBW(FILE *pFile)
{
    fprintf(pFile, "P4\n%ld %ld\n", width, height);
    
    long bytesperline = (width % 8 == 0 ? width / 8 : width / 8 + 1);
    unsigned char *pixelline_out = new unsigned char[bytesperline];

    for (long y = 0; y < height; ++y) {
        unsigned char *out_ptr = pixelline_out;
        GrayPixel *pixelline_in = ppGrayPixels[y];
        int bitnum = 7;                    // most significant bit first
        
        *out_ptr = 0;                      // clear the bits at the first byte position
        for (long x = 0; x < width; ++x)
        {
            if (pixelline_in[x] == 0)      // black is stored as '1' in the file!
            {
                *out_ptr |= (1 << bitnum); // put a bit
            }
            bitnum -= 1;                   // next bit is less significant
            if (bitnum < 0)
            {
                if (x < width - 1)
                {
                    bitnum = 7;                // highest bit first
                    out_ptr += 1;
                    *out_ptr = 0;              // clear the bits at the next byte position
                }                
            }   
        }
        fwrite(pixelline_out, sizeof(unsigned char), bytesperline, pFile);
    }
    delete[] pixelline_out;
}

void PamImage::_saveGray(FILE *pFile)
{
    fprintf(pFile, "P5\n%ld %ld\n255\n", width, height);
    for (long y = 0; y < height; ++y) {
        fwrite(ppGrayPixels[y], sizeof(GrayPixel), width, pFile);
    }
}

void PamImage::_saveRGB(FILE *pFile)
{
    fprintf(pFile, "P6\n%ld %ld\n255\n", width, height);
    unsigned char *pixelline_out = new unsigned char [3 * width];
    for (long y = 0; y < height; ++y) {
        unsigned char *out_ptr = pixelline_out; // points to element in output line array
        RGBPixel* pixelline_in = ppRGBPixels[y];
        
        for (long x = 0; x < width; ++x) {
            out_ptr[0] = pixelline_in[x].r;
            out_ptr[1] = pixelline_in[x].g;
            out_ptr[2] = pixelline_in[x].b;
            // ?? = ppRGBPixels[y][x].m;
            out_ptr += 3;
        }
        fwrite(pixelline_out, sizeof(unsigned char), 3 * width, pFile);
    }
    delete[] pixelline_out;
}

// Try to make a color image of an int image. Most significant byte is not stored!
void PamImage::_saveInt(FILE *pFile)
{
    bool datalost = false;
    fprintf(pFile, "P6\n%ld %ld\n255\n", width, height);
    unsigned char *pixelline_out = new unsigned char [3 * width];
    for (long y = 0; y < height; ++y) {
        unsigned char *out_ptr = pixelline_out; // points to element in output line array
        IntPixel* pixelline_in = ppIntPixels[y];
        
        for (long x = 0; x < width; ++x) {
            out_ptr[0] = (pixelline_in[x] >> 16) & 255;
            out_ptr[1] = (pixelline_in[x] >> 8) & 255;
            out_ptr[2] = pixelline_in[x] & 255;
            if ((pixelline_in[x] >> 24) & 255 > 0)
            {
                datalost = true;
            }
            // ?? = ppRGBPixels[y][x].m;
            out_ptr += 3;
        }
        fwrite(pixelline_out, sizeof(unsigned char), 3 * width, pFile);
    }
    delete[] pixelline_out;
    if (datalost)
    {
        std::cerr << "Warning: not all integer values were mapped to a unique color." << std::endl;
    }
}

// Save image in format regardless of file extension.
void PamImage::save(std::string _sFileName)
{
    FILE *pFile; // file pointer
    pFile = fopen(_sFileName.c_str(), "wb");
    if (pFile == NULL)
    {
         std::cerr << "Unable to open output file " << _sFileName << std::endl;
         throw 1;
    }

    if (iImageType == BW_IMAGE)
    {
        _saveBW(pFile);
    }
    else if (iImageType == RGB_IMAGE)
    {
        _saveRGB(pFile);
    }
    else if (iImageType == GRAY_IMAGE)
    {
        _saveGray(pFile);
    }
    else if (iImageType == INT_IMAGE)
    {
        _saveInt(pFile);
    }
    else
    {
        std::cerr << "Error (cannot write file): unknown image type: " << iImageType << "\n";
        throw 1;
    }
    fclose(pFile);
}

bool PamImage::isValid()
{
    bool bIsValid = 1;

    // if (strcmp(PamImage::sFileName.c_str(), "") == 0) bIsValid = 0;
    if (iImageType == NO_IMAGE) bIsValid = 0;
    
    return bIsValid;
}

std::string PamImage::getFileName()
{
    return PamImage::sFileName;
}

int PamImage::getWidth()
{
    if (!PamImage::isValid() ) {return -1;}
    return width;
}

int PamImage::getHeight()
{
    if (!PamImage::isValid() ) {return -1;}
    return height;
}

std::string PamImage::getFormat()
{
    std::string sReturnString;
    if (!PamImage::isValid() ) 
    {   
        sReturnString = "ERROR";
    }
    else if (iImageType == BW_IMAGE)
    {
        sReturnString = "BW";
    }
    else if (iImageType == GRAY_IMAGE)
    {
        sReturnString = "GRAY";
    }
    else if (iImageType == RGB_IMAGE)
    {
        sReturnString = "RGB";
    }
    else
    {
        sReturnString = "UNKNOWN";
    }

    return sReturnString;    
}

// Return grayscale/blackwhite image data as a 2D array
GrayPixel **PamImage::getGrayPixels()
{
    if ((iImageType != GRAY_IMAGE) && (iImageType != BW_IMAGE))
    {
        std::cerr << "Illegal pixel pointer requested: getGrayPixels() while type is "  << iImageType << std::endl;
        throw 1;
    }
    else
    {
        return ppGrayPixels;
    }
}

// Return color image data as a 2D array
RGBPixel **PamImage::getRGBPixels()
{
    if (iImageType != RGB_IMAGE)
    {
        std::cerr << "Illegal pixel pointer requested: getRGBPixels() while type is "  << iImageType << std::endl;
        throw 1;
    }
    else
    {
        return ppRGBPixels;
    }
}

// Return integer image data as a 2D array
IntPixel **PamImage::getIntPixels()
{
    if (iImageType != INT_IMAGE)
    {
        std::cerr << "Illegal pixel pointer requested: getIntPixels() while type is "  << iImageType << std::endl;
        throw 1;
    }
    else
    {
        return ppIntPixels;
    }
}

// return image type; NO_IMAGE = 0; BW_IMAGE = 1; GRAY_IMAGE = 2; RGB_IMAGE = 3
int PamImage::getImageType()
{
    return iImageType;
}

// Put a grayscale/bw pixel value in a grayscale/bw image
void PamImage::putPixel(long x, long y, int grayval)
{
    if ((iImageType != GRAY_IMAGE) && (iImageType != BW_IMAGE))
    {
        std::cerr << "Error: wrong pixel type: " << grayval << std::endl;
        throw 1;
    }
    else if ((x < 0) || (x >= width) || (y < 0) || (y >= height))
    {
        std::cerr << "Error: coordinates out of range: " << x << ", " << y << std::endl;
        throw 1;
    }
    else if ((grayval < 0) || (grayval > 255))
    {
        std::cerr << "Error: grayscale value out of range: " << grayval << std::endl;
        throw 1;
    }
    else
    {
        ppGrayPixels[y][x] = grayval;
    }
}

IntPixel PamImage::getPixelInt(int x, int y)
{
    if (iImageType != INT_IMAGE)
    {
        std::cerr << "Error: wrong pixel type; image is not Int" << std::endl;
        throw 1;
    }
    else if ((x < 0) || (x >= width) || (y < 0) || (y >= height))
    {
        std::cerr << "Error: coordinates out of range: " << x << ", " << y << std::endl;
        throw 1;
    }
    else
    {
        return ppIntPixels[y][x];
    }
}    

GrayPixel PamImage::getPixelGray(int x, int y)
{
    if ((iImageType != GRAY_IMAGE) && (iImageType != BW_IMAGE))
    {
        std::cerr << "Error: wrong pixel type; image is not grayscale or black/white" << std::endl;
        throw 1;
    }
    else if ((x < 0) || (x >= width) || (y < 0) || (y >= height))
    {
        std::cerr << "Error: coordinates out of range: " << x << ", " << y << std::endl;
        throw 1;
    }
    else
    {
        return ppGrayPixels[y][x];
    }
}

RGBPixel PamImage::getPixelRGB(float x, float y)
{    
  /* Uses linear interpolation to return an interpolated point.
   * Assumes that the image dimensions are [-0.5..width-0.5] x [-0.5..height-0.5]
   */

    long x1, x2, y1, y2;
    float w1, w2, w3, w4; // weights:   w1 w2
                          //            w3 w4
    float xd, yd;         // distances
    float r, g, b, m;     // return values as floats
    RGBPixel retpixel;    // return value
    
    // determine neighboring pixels
    x1 = (long) x;
    x2 = x1 + 1;
    y1 = (long) y;
    y2 = y1 + 1;
    
    // determine relative position of interpolation point
    xd = x - (float) x1;
    yd = y - (float) y1;
    
    // determine pixel weights
    w1 = (1 - xd) * (1 - yd);
    w2 = xd * (1 - yd);
    w3 = (1 - xd) * yd;
    w4 = xd * yd;

    // prepare output pixel
    r = 0.0;
    g = 0.0;
    b = 0.0;
    m = 0.0;
    
    // apply weights for pixels inside the image
    if ((y1 >= 0) && (y1 < height)) {
        // top row
        if ((x1 >= 0) && (x1 < width)) {
            // top-left
            r += w1 * ppRGBPixels[y1][x1].r;
            g += w1 * ppRGBPixels[y1][x1].g;
            b += w1 * ppRGBPixels[y1][x1].b;
            m += w1 * ppRGBPixels[y1][x1].m;
        }
        if ((x2 >= 0) && (x2 < width)) {
            // top-right
            r += w2 * ppRGBPixels[y1][x2].r;
            g += w2 * ppRGBPixels[y1][x2].g;
            b += w2 * ppRGBPixels[y1][x2].b;
            m += w2 * ppRGBPixels[y1][x2].m;
        }
    }
    if ((y2 >= 0) && (y2 < height)) {
        // bottom row
        if ((x1 >= 0) && (x1 < width)) {
            // bottom-left
            r += w3 * ppRGBPixels[y2][x1].r;
            g += w3 * ppRGBPixels[y2][x1].g;
            b += w3 * ppRGBPixels[y2][x1].b;
            m += w3 * ppRGBPixels[y2][x1].m;
        }
        if ((x2 >= 0) && (x2 < width)) {
            // bottom-right
            r += w4 * ppRGBPixels[y2][x2].r;
            g += w4 * ppRGBPixels[y2][x2].g;
            b += w4 * ppRGBPixels[y2][x2].b;
            m += w4 * ppRGBPixels[y2][x2].m;
        }
    }
    retpixel.r = (unsigned char) r;
    retpixel.g = (unsigned char) g;
    retpixel.b = (unsigned char) b;
    retpixel.m = (unsigned char) m;
    
    return retpixel;
}

int PamImage::get_minval()
{
    if (iImageType == INT_IMAGE) {
        int minval = ppIntPixels[0][0];
        for (long y = 0; y < height; ++y) {
            for (long x = 0; x < width; ++x) {
                int curval = (int) ppIntPixels[y][x];
                if (curval < minval) {
                    minval = curval;
                }
            }
        }
        return minval;        
    } else {
        std::cerr << "Error: get_minval() not implemented for image type " << getFormat() << std::endl;
        throw 1;
    }
}

int PamImage::get_maxval()
{
    if (iImageType == INT_IMAGE) {
        int maxval = ppIntPixels[0][0];
        for (long y = 0; y < height; ++y) {
            for (long x = 0; x < width; ++x) {
                int curval = (int) ppIntPixels[y][x];
                if (curval > maxval) {
                    maxval = curval;
                }
            }
        }
        return maxval;
    } else {
        std::cerr << "Error: get_maxval() not implemented for image type " << getFormat() << std::endl;
        throw 1;
    }
}

void PamImage::printAsciiArt()
{
    if (iImageType == BW_IMAGE)
    {
        for (long y = 0; y < height; ++y)
        {
            for (long x = 0; x < width; ++x)
            {
                GrayPixel p = ppGrayPixels[y][x];
                if (p == 0)
                {
                    std::cout << "#";
                }
                else
                {
                    std::cout << ".";
                }   
            }
            std::cout << std::endl;
        }
    }
    else if (iImageType == GRAY_IMAGE)
    {
        for (long y = 0; y < height; ++y)
        {
            for (long x = 0; x < width; ++x)
            {
                GrayPixel p = ppGrayPixels[y][x];
                if (p == 0) {
                    std::cout << "#";
                } else if ((p > 0) && (p < 127)) {
                    std::cout << "+";
                } else if ((p >= 127) && (p < 255)) {
                    std::cout << "-";
                }
                else
                {
                    std::cout << ".";
                }   
            }
            std::cout << std::endl;
        }
    }
    else
    {
        std::cerr << "Warning: Ascii art for image type " << iImageType << " not implemented yet." << std::endl;
    }
}

// Should only be called from convert. Conversion types are known.
PamImage* PamImage::_convertBWToGray()
{
    PamImage* out_im = new PamImage(GRAY_IMAGE, width, height);
    GrayPixel **pixels_out = out_im->getGrayPixels();
    for (long y = 0; y < height; ++y)
    {
        for (long x = 0; x < width; ++x)
        {
            if (ppGrayPixels[y][x] == 0)
            {
                pixels_out[y][x] = 0;
            }
            else
            {
                pixels_out[y][x] = 255;
            }
        }
    }
    return out_im;
}

// Should only be called from convert. Conversion types are known.
// Applies scaling of the ints to [0..255]
PamImage* PamImage::_convertIntToGray()
{
    PamImage* out_im = new PamImage(GRAY_IMAGE, width, height);
    GrayPixel **pixels_out = out_im->getGrayPixels();
    
    int minval = get_minval();
    int maxval = get_maxval();
    //if (maxval == 0) maxval = 1;
    //float scale = 255.0 / maxval;
    int range = maxval - minval;
    if (range == 0) {
        range = 1;
    } 
    float scale = 255.0 / (float) (range);
    
    for (long y = 0; y < height; ++y)
    {
        for (long x = 0; x < width; ++x)
        {
            pixels_out[y][x] = (GrayPixel) round(scale * (float) (ppIntPixels[y][x] + minval));
            //std::cerr << "ppIntPixels[y][x]" << "->" << (int) pixels_out[y][x] << " ";
        }
    }
    return out_im;
}

// Should only be called from convert. Conversion types are known.
PamImage* PamImage::_convertBWToRGB()
{
    PamImage* out_im = new PamImage(RGB_IMAGE, width, height);
    RGBPixel **pixels_out = out_im->getRGBPixels();
    for (long y = 0; y < height; ++y)
    {
        for (long x = 0; x < width; ++x)
        {
        	RGBPixel& p = pixels_out[y][x];
            if (ppGrayPixels[y][x] == 0)
            {
                p.r = 0;
                p.g = 0;
                p.b = 0;
                p.m = 0;
            }
            else
            {
                p.r = 255;
                p.g = 255;
                p.b = 255;
                p.m = 255;
            }
        }
    }
    return out_im;
}

// Should only be called from convert. Conversion types are known.
PamImage* PamImage::_convertGrayToRGB()
{
    PamImage* out_im = new PamImage(RGB_IMAGE, width, height);
    RGBPixel **pixels_out = out_im->getRGBPixels();
    for (long y = 0; y < height; ++y)
    {
        for (long x = 0; x < width; ++x)
        {
            pixels_out[y][x].r = ppGrayPixels[y][x];
            pixels_out[y][x].g = ppGrayPixels[y][x];
            pixels_out[y][x].b = ppGrayPixels[y][x];
        }
    }
    return out_im;
}

// Should only be called from convert. Conversion types are known.
PamImage* PamImage::_convertRGBToGray()
{
    PamImage* out_im = new PamImage(GRAY_IMAGE, width, height);
    GrayPixel **pixels_out = out_im->getGrayPixels();
    for (long y = 0; y < height; ++y) {
        for (long x = 0; x < width; ++x) {
            RGBPixel &p = ppRGBPixels[y][x];
            pixels_out[y][x] = ((int) p.r + (int) p.g + (int) p.b) / 3;
        }
    }
    return out_im;
}

// Use a fixed threshold of 128 
PamImage* PamImage::_convertGrayToBW()
{
    PamImage* out_im = new PamImage(BW_IMAGE, width, height);
    GrayPixel **pixels_out = out_im->getGrayPixels();
    for (long y = 0; y < height; ++y)
    {
        for (long x = 0; x < width; ++x)
        {
            if (ppGrayPixels[y][x] < 128)
            {
                pixels_out[y][x] = 0;
            }
            else
            {
                pixels_out[y][x] = 1;
            }
        }
    }
    return out_im;
}

// Convert to new image type
PamImage* PamImage::convert(int _dstImageType)
{
    if (_dstImageType == iImageType) {
        return new PamImage(*this);
    }
    switch (iImageType) {
        case BW_IMAGE:
        {
            switch (_dstImageType) {
                case GRAY_IMAGE: { return _convertBWToGray(); break; }
                case RGB_IMAGE:  { return _convertBWToRGB(); break; } 
                case INT_IMAGE:  { std::cerr << "Error: conversion BW->Int not implemented" << std::endl; throw 1; break; }
            }
            break;            
        }
        case GRAY_IMAGE:
        {
            switch (_dstImageType) {
                case BW_IMAGE:   { return _convertGrayToBW(); break; } 
                case RGB_IMAGE:  { return _convertGrayToRGB(); break; } 
                case INT_IMAGE:  { std::cerr << "Error: conversion Gray->Int not implemented" << std::endl; throw 1; break; }
            }
            break;            
        }
        case RGB_IMAGE:
        {
            switch (_dstImageType) {
                case BW_IMAGE:   { std::cerr << "Error: conversion RGB->BW not implemented" << std::endl; throw 1; break; } 
                case GRAY_IMAGE: { return _convertRGBToGray(); break; } 
                case INT_IMAGE:  { std::cerr << "Error: conversion RGB->Int not implemented" << std::endl; throw 1; break; }
            }
            break;            
        }
        case INT_IMAGE:
        {
            switch (_dstImageType) {
                case BW_IMAGE:   { std::cerr << "Error: conversion Int->BW not implemented" << std::endl; throw 1; break; } 
                case GRAY_IMAGE: {return _convertIntToGray(); break; } 
                case RGB_IMAGE:  { std::cerr << "Error: conversion Int->RGB not implemented" << std::endl; throw 1; break; } 
            }
            break;            
        }
    }
    std::cerr << "Error: conversion failed." << std::endl;
    throw 1;
}
