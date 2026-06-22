#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <dirent.h>
#include <omp.h>

#include "openmp_preprocessing.c"

#define DEFAULT_WIDTH 320
#define DEFAULT_HEIGHT 240
#define DEFAULT_REPETITIONS 30
#define FALLBACK_IMAGE_COUNT 64
#define THREAD_COUNT_OPTIONS 4

static int has_image_extension(const char* filename) {
    const char* dot = strrchr(filename, '.');
    char ext[16];
    size_t i;

    if (dot == NULL) {
        return 0;
    }

    for (i = 0; dot[i] != '\0' && i < sizeof(ext) - 1u; ++i) {
        ext[i] = (char)tolower((unsigned char)dot[i]);
    }
    ext[i] = '\0';

    return strcmp(ext, ".jpg") == 0
        || strcmp(ext, ".jpeg") == 0
        || strcmp(ext, ".png") == 0
        || strcmp(ext, ".bmp") == 0
        || strcmp(ext, ".webp") == 0;
}

static size_t count_images_in_dir(const char* folder) {
    DIR* dir = opendir(folder);
    struct dirent* entry;
    size_t count = 0u;

    if (dir == NULL) {
        return 0u;
    }

    while ((entry = readdir(dir)) != NULL) {
        if (has_image_extension(entry->d_name)) {
            ++count;
        }
    }

    closedir(dir);
    return count;
}

static size_t count_dataset_images(void) {
    const char* folders[] = {
        "../../datasets/train/no_yawn",
        "../../datasets/train/yawn",
        "../../datasets/validation/no_yawn",
        "../../datasets/validation/yawn",
        "../../datasets/test/no_yawn",
        "../../datasets/test/yawn",
    };
    size_t total = 0u;
    size_t i;

    for (i = 0u; i < sizeof(folders) / sizeof(folders[0]); ++i) {
        total += count_images_in_dir(folders[i]);
    }

    return total;
}

static int make_deterministic_image(RGBImage* image, int width, int height, size_t seed) {
    int y;
    int x;
    image->width = width;
    image->height = height;
    image->pixels = (uint8_t*)malloc((size_t)width * (size_t)height * 3u * sizeof(uint8_t));

    if (image->pixels == NULL) {
        return 0;
    }

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            const size_t index = ((size_t)y * (size_t)width + (size_t)x) * 3u;
            image->pixels[index] = (uint8_t)((x + (int)seed * 13) % 256);
            image->pixels[index + 1u] = (uint8_t)((y * 2 + (int)seed * 7) % 256);
            image->pixels[index + 2u] = (uint8_t)((x + y + (int)seed * 3) % 256);
        }
    }

    return 1;
}

static RGBImage* build_workload(size_t image_count, int width, int height) {
    RGBImage* images = (RGBImage*)calloc(image_count, sizeof(RGBImage));
    size_t i;

    if (images == NULL) {
        return NULL;
    }

    for (i = 0u; i < image_count; ++i) {
        if (!make_deterministic_image(&images[i], width, height, i)) {
            size_t j;
            for (j = 0u; j < i; ++j) {
                free(images[j].pixels);
            }
            free(images);
            return NULL;
        }
    }

    return images;
}

static void free_workload(RGBImage* images, size_t image_count) {
    size_t i;
    if (images == NULL) {
        return;
    }
    for (i = 0u; i < image_count; ++i) {
        free(images[i].pixels);
    }
    free(images);
}

static double measure_serial_ms(const RGBImage* images, size_t image_count, int repetitions, double* checksum) {
    float* output = (float*)malloc((size_t)OUTPUT_FEATURES * sizeof(float));
    double start;
    double end;
    int rep;

    if (output == NULL) {
        return -1.0;
    }

    start = omp_get_wtime();
    for (rep = 0; rep < repetitions; ++rep) {
        size_t i;
        for (i = 0u; i < image_count; ++i) {
            if (!preprocess_serial(&images[i], output)) {
                free(output);
                return -1.0;
            }
            *checksum += output[(rep + images[i].width + images[i].height) % OUTPUT_FEATURES];
        }
    }
    end = omp_get_wtime();

    free(output);
    return (end - start) * 1000.0;
}

static double measure_openmp_ms(const RGBImage* images, size_t image_count, int repetitions, int threads, double* checksum) {
    float* output = (float*)malloc((size_t)OUTPUT_FEATURES * sizeof(float));
    double start;
    double end;
    int rep;

    if (output == NULL) {
        return -1.0;
    }

    start = omp_get_wtime();
    for (rep = 0; rep < repetitions; ++rep) {
        size_t i;
        for (i = 0u; i < image_count; ++i) {
            if (!preprocess_openmp(&images[i], output, threads)) {
                free(output);
                return -1.0;
            }
            *checksum += output[(rep + images[i].width + images[i].height) % OUTPUT_FEATURES];
        }
    }
    end = omp_get_wtime();

    free(output);
    return (end - start) * 1000.0;
}

static void write_csv_row(FILE* csv, const char* mode, int threads, size_t processed_images, double time_ms, double speedup) {
    fprintf(csv, "%s,%d,%zu,%.3f,%.3f\n", mode, threads, processed_images, time_ms, speedup);
}

int main(int argc, char** argv) {
    const int width = argc > 1 ? atoi(argv[1]) : DEFAULT_WIDTH;
    const int height = argc > 2 ? atoi(argv[2]) : DEFAULT_HEIGHT;
    const int repetitions = argc > 3 ? atoi(argv[3]) : DEFAULT_REPETITIONS;
    const int thread_counts[THREAD_COUNT_OPTIONS] = {1, 2, 4, 8};
    const size_t dataset_images = count_dataset_images();
    const size_t image_count = dataset_images > 0u ? dataset_images : FALLBACK_IMAGE_COUNT;
    const size_t processed_images = image_count * (size_t)repetitions;
    RGBImage* images = NULL;
    FILE* csv = NULL;
    double checksum = 0.0;
    double serial_ms;
    int i;

    if (width <= 1 || height <= 1 || repetitions <= 0) {
        fprintf(stderr, "Uso: benchmark_openmp.exe [width height repetitions]\n");
        return 1;
    }

    images = build_workload(image_count, width, height);
    if (images == NULL) {
        fprintf(stderr, "No fue posible reservar memoria para la carga de trabajo.\n");
        return 1;
    }

    csv = fopen("../../metrics/openmp_benchmark.csv", "w");
    if (csv == NULL) {
        fprintf(stderr, "No fue posible abrir ../../metrics/openmp_benchmark.csv\n");
        free_workload(images, image_count);
        return 1;
    }

    fprintf(csv, "mode,threads,images,time_ms,speedup\n");

    serial_ms = measure_serial_ms(images, image_count, repetitions, &checksum);
    if (serial_ms < 0.0) {
        fprintf(stderr, "Fallo el benchmark serial.\n");
        fclose(csv);
        free_workload(images, image_count);
        return 1;
    }

    write_csv_row(csv, "serial", 1, processed_images, serial_ms, 1.0);

    for (i = 0; i < THREAD_COUNT_OPTIONS; ++i) {
        const int threads = thread_counts[i];
        const double openmp_ms = measure_openmp_ms(images, image_count, repetitions, threads, &checksum);
        const double speedup = openmp_ms > 0.0 ? serial_ms / openmp_ms : 0.0;

        if (openmp_ms < 0.0) {
            fprintf(stderr, "Fallo el benchmark OpenMP con %d hilos.\n", threads);
            fclose(csv);
            free_workload(images, image_count);
            return 1;
        }

        write_csv_row(csv, "openmp", threads, processed_images, openmp_ms, speedup);
    }

    fclose(csv);

    printf("Imagenes detectadas en datasets/: %zu\n", dataset_images);
    printf("Imagenes procesadas por modo: %zu\n", processed_images);
    printf("Tamano de entrada sintetica: %dx%d\n", width, height);
    printf("Salida del preprocesamiento: 80x80 = 6400 caracteristicas\n");
    printf("CSV generado: ../../metrics/openmp_benchmark.csv\n");
    printf("Checksum: %.6f\n", checksum);

    free_workload(images, image_count);
    return 0;
}
