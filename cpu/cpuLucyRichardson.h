
# pragma once

int cpuLucyRichardson(unsigned W, unsigned H, int num_iter, unsigned char *image_input, unsigned char *image_output);

void CpuLucyRichIteration(const unsigned char * c, unsigned char *g, unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const unsigned W, const unsigned H);

void updatePSF(const unsigned char *c, unsigned char *g, const unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const unsigned W, const unsigned H);

void updateUnderlyingImg(const unsigned char *c, const unsigned char *g, unsigned char *f, unsigned char *tmp1, unsigned char *tmp2, const unsigned W, const unsigned H);

void convolve(const unsigned char *A, const unsigned char *B, unsigned char *C, const unsigned W, const unsigned H);

void elementWiseDivision(const unsigned char *A, const unsigned char *B, unsigned char *C, const unsigned W, const unsigned H);

void elementWiseMultiplication(const unsigned char *A, const unsigned char *B, unsigned char *C, const unsigned W, const unsigned H);