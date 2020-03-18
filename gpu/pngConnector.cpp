
#include <vector>
#include <iostream>
#include <png++/png.hpp>


#include "pngConnector.hpp"



Image loadImage(const std::string &filename)
{
    png::image<png::rgb_pixel> in_image(filename);
    Image imageMatrix(3, Matrix(in_image.get_height(), Array(in_image.get_width())));

    int h, w;
    for (h = 0; h < in_image.get_height(); h++)
    {
        for (w = 0; w < in_image.get_width(); w++)
        {
            imageMatrix[0][h][w] = in_image[h][w].red;
            imageMatrix[1][h][w] = in_image[h][w].green;
            imageMatrix[2][h][w] = in_image[h][w].blue;
        }
    }
    return imageMatrix;
}


void saveImage(Image &image, const std::string &filename)
{
    assert(image.size() == 3);

    int height = image[0].size();
    int width = image[0][0].size();
    int x, y;

    png::image<png::rgb_pixel> imageFile(width, height);

    for (y = 0; y < height; y++)
    {
        for (x = 0; x < width; x++)
        {
            imageFile[y][x].red = image[0][y][x];
            imageFile[y][x].green = image[1][y][x];
            imageFile[y][x].blue = image[2][y][x];
        }
    }
    imageFile.write(filename);
}