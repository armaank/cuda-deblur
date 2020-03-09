#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include <png++/png.hpp>

typedef std::vector<double> Array;
typedef std::vector<Array> Matrix;
typedef std::vector<Matrix> Image;

Matrix gaussian(int height, int width, double sigma)
{
    Matrix kernel(height, Array(width));
    double sum = 0.0;
    int i, j;

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            kernel[i][j] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            sum += kernel[i][j];
        }
    }

    for (i = 0; i < height; i++)
    {
        for (j = 0; j < width; j++)
        {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}
Matrix create(int height, int width)
{
    return Matrix(height, Array(width, 0));
}
Image loadImage(const char *filename)
{
    png::image<png::rgb_pixel> image(filename);
    Image imageMatrix(3, Matrix(image.get_height(), Array(image.get_width())));

    int h, w;
    for (h = 0; h < image.get_height(); h++)
    {
        for (w = 0; w < image.get_width(); w++)
        {
            imageMatrix[0][h][w] = image[h][w].red;
            imageMatrix[1][h][w] = image[h][w].green;
            imageMatrix[2][h][w] = image[h][w].blue;
        }
    }

    return imageMatrix;
}

void saveImage(Image &image, const char *filename)
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

Image conv(Image &image, Matrix &filter)
{
    assert(image.size() == 3 && filter.size() != 0);

    int height = image[0].size();
    int width = image[0][0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();
    int newImageHeight = height; // - filterHeight + 1;
    int newImageWidth = width;   // - filterWidth + 1;
    int d, i, j, h, w;

    Image newImage(3, Matrix(newImageHeight, Array(newImageWidth)));
    /* lazy way to pad w/ zeros, probably a much better way to do this */
    for (d = 0; d < 3; d++)
    {
        for (i = 0; i < newImageHeight; i++)
        {
            for (j = 0; j < newImageWidth; j++)
            {
                newImage[d][i][j] += 0;
            }
        }
    }

    newImageHeight = height - filterHeight + 1;
    newImageWidth = width - filterWidth + 1;
    for (d = 0; d < 3; d++)
    {
        for (i = 0; i < newImageHeight; i++)
        {
            for (j = 0; j < newImageWidth; j++)
            {
                for (h = i; h < i + filterHeight; h++)
                {
                    for (w = j; w < j + filterWidth; w++)
                    {
                        newImage[d][i][j] += filter[h - i][w - j] * image[d][h][w];
                    }
                }
            }
        }
    }

    return newImage;
}

Matrix multiply(const Matrix &a, const Matrix &b)
{
    /* make sure that height and width match */
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Matrix result = create(a.size(), a[0].size());
    for (int i = 0; i < result.size(); i++)
    {
        for (int j = 0; j < result[0].size(); j++)
        {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

Matrix divide(const Matrix &a, const Matrix &b)
{
    /* make sure that height and width match */
    assert(a.size() == b.size() && a[0].size() == b[0].size());
    Matrix result = create(a.size(), a[0].size());
    for (int i = 0; i < result.size(); i++)
    {
        for (int j = 0; j < result[0].size(); j++)
        {
            result[i][j] = a[i][j] / b[i][j];
        }
    }
    return result;
}

Image rlDeconv(Image &image, Matrix &filter, int n_iter)
{
    /*
     initilize final, deblurred image
     initilized to input, blurred image
    
     */
    Image im_deconv = image;
    Image rel_blur = image;
    int filt_length = filter.size();
    int filt_width = filter[0].size();
    /* compute and store mirrored psf */
    Matrix filter_m(filt_length, Array(filt_width));
    for (int i = 0; i < filt_length; i++)
    {
        for (int j = 0; j < filt_width; j++)
        {
            filter_m[i][j] = filter[j][i];
        }
    }

    /* perform lucy iterations */
    for (int i = 0; i < n_iter; i++)
    {
        std::cout << "iter number: " << i << std::endl;

        /* filter target image by psf */

        Image tmp1 = conv(im_deconv, filter);

        /* perform true division accross channels to compute relative blur */
        for (int d = 0; d < 3; d++)
        {
            rel_blur[d] = divide(image[d], tmp1[d]);
        }

        /* filter blur by psf */
        Image tmp2 = conv(rel_blur, filter_m);

        /* perform multiplication of tmp2 and im_deconv to update the deblurred image */
        for (int d = 0; d < 3; d++)
        {
            im_deconv[d] = multiply(tmp2[d], im_deconv[d]);
        }
    }

    return im_deconv;
}

int main(int argc, char **argv)
{
    const char *input_file = argv[1];
    const char *output_file = argv[2];
    int num_iter = 10;
    if (argc <= 2)
    {
        std::cerr << "error: specify input and output files" << std::endl;
        return -1;
    }

    Matrix filter = gaussian(3, 3, 1);
    std::cout << "Loading image..." << std::endl;
    Image image = loadImage(input_file);
    std::cout << "running lucy iterations..." << std::endl;
    Image newImage = rlDeconv(image, filter, 5);
    std::cout << "Saving image..." << std::endl;
    saveImage(newImage, output_file);
    std::cout << "Done!" << std::endl;
}
