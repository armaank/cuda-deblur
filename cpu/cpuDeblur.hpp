#pragma once

Image rlDeconv(const Image &image, const Matrix &filter, const int n_iter);
Image cpuDeblur(const Image &imgae, const Matrix &filter);
