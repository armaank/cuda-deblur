#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include "png_lib/lodepng.h"
#include "cpuLucyRichardson.h"

#define MAX_PIXEL 255

int main(int argc, char **argv)
{
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int num_iter = 10;
    if (argc <= 2)
    {
        std::cout << "error: specify input and output files" << std::endl;
        return -1;
    }

    /* input image */
    std::vector<unsigned char> in_image;
    unsigned int width, height;

    /* decode .png input, error check */
    unsigned error = lodepng::decode(in_image, width, height, input_file);
    if (error)
    {
        std::cout << "error decoding input .png" << error << ": " << lodepng_error_text(error) << std::endl;
    }
    /* the png is now loaded in "image" */

    /* convert data from rgba to rgb, init temporary arrays for processing */
    unsigned char *input_image_proc = new unsigned char[(in_image.size() * 3) / 4];
    unsigned char *output_image_proc = new unsigned char[(in_image.size() * 3) / 4];
    int pointer = 0;
    for (int i = 0; i < in_image.size(); ++i)
    {
        if ((i + 1) % 4 != 0)
        {
            input_image_proc[pointer] = in_image.at(i);
            output_image_proc[pointer] = MAX_PIXEL;
            pointer++;
        }
    }

    /* fcn. kernel/lucy richardson */
    // cpuLucyRichardson(width, height, num_iter, input_image_proc, output_image_proc);

    /* initialize output .png */

    std::vector<unsigned char> out_image;
    for (int i = 0; i < in_image.size(); ++i)
    {
        out_image.push_back(input_image_proc[i]); // changed from output_image_proc
        if ((i + 1) % 3 == 0)
        {
            out_image.push_back(MAX_PIXEL);
        }
    }

    /* encode image as .png, error check */
    error = lodepng::encode(output_file, out_image, width, height);
    if (error)
    {
        std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }

    /* clean-up temporary vars for processing */
    delete[] input_image_proc;
    delete[] output_image_proc;

    return 0;
}