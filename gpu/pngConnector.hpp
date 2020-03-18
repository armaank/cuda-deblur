

#pragma once


using Array  = std::vector<double>;
using Matrix = std::vector<Array>;
using Image  = std::vector<Matrix>;

Image loadImage(const std::string &filename);
void saveImage(Image &image, const std::string &filename);