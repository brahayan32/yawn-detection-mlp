#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define OUTPUT_WIDTH 80
#define OUTPUT_HEIGHT 80
#define OUTPUT_FEATURES (OUTPUT_WIDTH * OUTPUT_HEIGHT)

typedef struct {
    int width;
    int height;
    uint8_t* pixels;
} RGBImage;

static const float GAUSSIAN_3X3[3][3] = {
    {1.0f, 2.0f, 1.0f},
    {2.0f, 4.0f, 2.0f},
    {1.0f, 2.0f, 1.0f},
};

static int clamp_int(int value, int low, int high) {
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

static int allocate_work_buffers(int width, int height, float** gray, float** blurred, float** resized) {
    const size_t input_size = (size_t)width * (size_t)height;
    *gray = (float*)malloc(input_size * sizeof(float));
    *blurred = (float*)malloc(input_size * sizeof(float));
    *resized = (float*)malloc((size_t)OUTPUT_FEATURES * sizeof(float));

    if (*gray == NULL || *blurred == NULL || *resized == NULL) {
        free(*gray);
        free(*blurred);
        free(*resized);
        *gray = NULL;
        *blurred = NULL;
        *resized = NULL;
        return 0;
    }

    return 1;
}

void rgb_to_grayscale_serial(const RGBImage* image, float* gray) {
    int y;
    int x;
    for (y = 0; y < image->height; ++y) {
        for (x = 0; x < image->width; ++x) {
            const size_t rgb_index = ((size_t)y * (size_t)image->width + (size_t)x) * 3u;
            const float r = (float)image->pixels[rgb_index];
            const float g = (float)image->pixels[rgb_index + 1u];
            const float b = (float)image->pixels[rgb_index + 2u];
            gray[(size_t)y * (size_t)image->width + (size_t)x] = 0.299f * r + 0.587f * g + 0.114f * b;
        }
    }
}

void rgb_to_grayscale_openmp(const RGBImage* image, float* gray) {
    const int total = image->width * image->height;
    int i;
#pragma omp parallel for
    for (i = 0; i < total; ++i) {
        const size_t rgb_index = (size_t)i * 3u;
        const float r = (float)image->pixels[rgb_index];
        const float g = (float)image->pixels[rgb_index + 1u];
        const float b = (float)image->pixels[rgb_index + 2u];
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

void gaussian_blur_serial(const float* gray, int width, int height, float* blurred) {
    int y;
    int x;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            float acc = 0.0f;
            int ky;
            int kx;
            for (ky = -1; ky <= 1; ++ky) {
                for (kx = -1; kx <= 1; ++kx) {
                    const int yy = clamp_int(y + ky, 0, height - 1);
                    const int xx = clamp_int(x + kx, 0, width - 1);
                    acc += gray[(size_t)yy * (size_t)width + (size_t)xx] * GAUSSIAN_3X3[ky + 1][kx + 1];
                }
            }
            blurred[(size_t)y * (size_t)width + (size_t)x] = acc / 16.0f;
        }
    }
}

void gaussian_blur_openmp(const float* gray, int width, int height, float* blurred) {
    int y;
#pragma omp parallel for
    for (y = 0; y < height; ++y) {
        int x;
        for (x = 0; x < width; ++x) {
            float acc = 0.0f;
            int ky;
            int kx;
            for (ky = -1; ky <= 1; ++ky) {
                for (kx = -1; kx <= 1; ++kx) {
                    const int yy = clamp_int(y + ky, 0, height - 1);
                    const int xx = clamp_int(x + kx, 0, width - 1);
                    acc += gray[(size_t)yy * (size_t)width + (size_t)xx] * GAUSSIAN_3X3[ky + 1][kx + 1];
                }
            }
            blurred[(size_t)y * (size_t)width + (size_t)x] = acc / 16.0f;
        }
    }
}

void resize_bilinear_serial(const float* input, int width, int height, float* resized) {
    const float x_scale = (float)(width - 1) / (float)(OUTPUT_WIDTH - 1);
    const float y_scale = (float)(height - 1) / (float)(OUTPUT_HEIGHT - 1);
    int y;

    for (y = 0; y < OUTPUT_HEIGHT; ++y) {
        const float src_y = (float)y * y_scale;
        const int y0 = (int)floorf(src_y);
        const int y1 = clamp_int(y0 + 1, 0, height - 1);
        const float wy = src_y - (float)y0;
        int x;

        for (x = 0; x < OUTPUT_WIDTH; ++x) {
            const float src_x = (float)x * x_scale;
            const int x0 = (int)floorf(src_x);
            const int x1 = clamp_int(x0 + 1, 0, width - 1);
            const float wx = src_x - (float)x0;
            const float top = input[(size_t)y0 * (size_t)width + (size_t)x0] * (1.0f - wx)
                + input[(size_t)y0 * (size_t)width + (size_t)x1] * wx;
            const float bottom = input[(size_t)y1 * (size_t)width + (size_t)x0] * (1.0f - wx)
                + input[(size_t)y1 * (size_t)width + (size_t)x1] * wx;
            resized[(size_t)y * (size_t)OUTPUT_WIDTH + (size_t)x] = top * (1.0f - wy) + bottom * wy;
        }
    }
}

void resize_bilinear_openmp(const float* input, int width, int height, float* resized) {
    const float x_scale = (float)(width - 1) / (float)(OUTPUT_WIDTH - 1);
    const float y_scale = (float)(height - 1) / (float)(OUTPUT_HEIGHT - 1);
    int y;

#pragma omp parallel for
    for (y = 0; y < OUTPUT_HEIGHT; ++y) {
        const float src_y = (float)y * y_scale;
        const int y0 = (int)floorf(src_y);
        const int y1 = clamp_int(y0 + 1, 0, height - 1);
        const float wy = src_y - (float)y0;
        int x;

        for (x = 0; x < OUTPUT_WIDTH; ++x) {
            const float src_x = (float)x * x_scale;
            const int x0 = (int)floorf(src_x);
            const int x1 = clamp_int(x0 + 1, 0, width - 1);
            const float wx = src_x - (float)x0;
            const float top = input[(size_t)y0 * (size_t)width + (size_t)x0] * (1.0f - wx)
                + input[(size_t)y0 * (size_t)width + (size_t)x1] * wx;
            const float bottom = input[(size_t)y1 * (size_t)width + (size_t)x0] * (1.0f - wx)
                + input[(size_t)y1 * (size_t)width + (size_t)x1] * wx;
            resized[(size_t)y * (size_t)OUTPUT_WIDTH + (size_t)x] = top * (1.0f - wy) + bottom * wy;
        }
    }
}

void normalize_flatten_serial(const float* resized, float* output) {
    int i;
    for (i = 0; i < OUTPUT_FEATURES; ++i) {
        output[i] = resized[i] / 255.0f;
    }
}

void normalize_flatten_openmp(const float* resized, float* output) {
    int i;
#pragma omp parallel for
    for (i = 0; i < OUTPUT_FEATURES; ++i) {
        output[i] = resized[i] / 255.0f;
    }
}

int preprocess_serial(const RGBImage* image, float* output) {
    float* gray = NULL;
    float* blurred = NULL;
    float* resized = NULL;

    if (!allocate_work_buffers(image->width, image->height, &gray, &blurred, &resized)) {
        return 0;
    }

    rgb_to_grayscale_serial(image, gray);
    gaussian_blur_serial(gray, image->width, image->height, blurred);
    resize_bilinear_serial(blurred, image->width, image->height, resized);
    normalize_flatten_serial(resized, output);

    free(gray);
    free(blurred);
    free(resized);
    return 1;
}

int preprocess_openmp(const RGBImage* image, float* output, int threads) {
    float* gray = NULL;
    float* blurred = NULL;
    float* resized = NULL;

#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif

    if (!allocate_work_buffers(image->width, image->height, &gray, &blurred, &resized)) {
        return 0;
    }

    rgb_to_grayscale_openmp(image, gray);
    gaussian_blur_openmp(gray, image->width, image->height, blurred);
    resize_bilinear_openmp(blurred, image->width, image->height, resized);
    normalize_flatten_openmp(resized, output);

    free(gray);
    free(blurred);
    free(resized);
    return 1;
}
