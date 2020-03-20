/**
 * pngConnector.cpp - Header file for the functions 
 * that call the png++ wrapper.
 */

#pragma once

#include "ops.hpp"


Image loadImage(const std::string &filename);
void  saveImage(const Image &image, const std::string &filename);