
# pragma once

int cpuLucyRichardson(const int W, const int H, const int num_iter, unsigned char *image_input, unsigned char *image_output);

void CpuLucyRichIteration(const unsigned char * c, unsigned char *g, unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const int W, const int H);

void updatePSF(const unsigned char *c, unsigned char *g, const unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const int W, const int H);

void updateUnderlyingImg(const unsigned char *c, const unsigned char *g, unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const int W, const int H);

void convolve(const unsigned char *A, const unsigned char *B, unsigned char *C, const int W, const int H);

void elementWiseDivision(const unsigned char *A, const unsigned char *B, unsigned char *C, const int W, const int H);

void elementWiseMultiplication(const unsigned char *A, const unsigned char *B, unsigned char *C, const int W, const int H);