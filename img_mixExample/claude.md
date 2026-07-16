Inside of the main.cu add a function

int readJPEG(const std::string filename, uchar* r, uchar* g, uchar* b, int& width, int& height)

which should read a JPEG RGB file
The parameters are;
filename: name of the image file.
r: Pointer to a memory location which contains the red color channel date
g: Pointer to a memory location which contains the green color channel date
b: Pointer to a memory location which contains the blue color channel date

width: Reference to a local variable, where the width of the image shall be stored in
height: Reference to a local variable, where the height of the image shall be stored in.

The function should return 0 on success, -1 otherwise.

New 001:

Add a function

int saveJPEG(const std::string filename, uchar* r, uchar* g, uchar* b, const int width, const int height, const int quality=90)

which encodes image RGB data to JPEG and writes the compressed data into a standard JPEG file.

The parameters are:
filename: Name of the output file
r: Pointer to the red data channel
g: Pointer to the green data channel
b: Pointer to the blue data channel
width: Image width
height: Image height
quality: JPEG compression quality

The function should return 0 on success, -1 otherwise.

New 002:

The executable shall be stored inside a subfolder /exe relative to the main project path

New 003:

A CUDA kernel should be used to perform processing on the image data like:
toGrayScale(uchar* r, uchar* g, uchar* b, int width, int height)

Device memory shall be allocated for r1, g1, b1 and r2, g2, b2
and the image data shall be copied to the device
The output of the processing is in r1 and r2 respectively and shall be copied back to the host memory

New 004:
Add two global const variables

float d_p1
float d_p2

which should take float constants p1 and p2 from a host variable