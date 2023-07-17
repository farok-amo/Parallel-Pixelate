#include <stdlib.h>
#include <string.h>

#include "openmp.h"

#include "helper.h"


Image omp_input_image;
Image omp_output_image;

unsigned int omp_TILES_X, omp_TILES_Y;
unsigned long long* omp_mosaic_sum;
unsigned char* omp_mosaic_value;

void openmp_begin(const Image *input_image) {
    omp_TILES_X = input_image->width / TILE_SIZE;
    omp_TILES_Y = input_image->height / TILE_SIZE;

    // Allocate buffer for calculating the sum of each tile mosaic
    omp_mosaic_sum = (unsigned long long*)malloc(omp_TILES_X * omp_TILES_Y * input_image->channels * sizeof(unsigned long long));

    // Allocate buffer for storing the output pixel value of each tile
    omp_mosaic_value = (unsigned char*)malloc(omp_TILES_X * omp_TILES_Y * input_image->channels * sizeof(unsigned char));

    // Allocate copy of input image
    omp_input_image = *input_image;
    omp_input_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
    memcpy(omp_input_image.data, input_image->data, input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));

    // Allocate output image
    omp_output_image = *input_image;
    omp_output_image.data = (unsigned char*)malloc(input_image->width * input_image->height * input_image->channels * sizeof(unsigned char));
}
void openmp_stage1() {
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_tile_sum(input_image, mosaic_sum);
    // Reset sum memory to 0
    memset(omp_mosaic_sum, 0, omp_TILES_X * omp_TILES_Y * omp_input_image.channels * sizeof(unsigned long long));

    int t, p_x, p_y;
 
#pragma omp parallel for private(t, p_x, p_y) 
    for (t = 0; t < omp_TILES_X * omp_TILES_Y; ++t) {
        // get original loop indecies to traverse the image 
        const int t_x = t % omp_TILES_X;
        const int t_y = t / omp_TILES_X;

        const unsigned int tile_index = t * omp_input_image.channels;
        const unsigned int tile_offset = (t_y * omp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * omp_input_image.channels;

        // For each pixel within the tile
        for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
            for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                // For each colour channel
                const unsigned int pixel_offset = (p_y * omp_input_image.width + p_x) * omp_input_image.channels;

                // Avoiding loop usage since channels are always 3 (R, G, B), improves performence and code coherency 
                omp_mosaic_sum[tile_index] += omp_input_image.data[tile_offset + pixel_offset];
                omp_mosaic_sum[tile_index + 1] += omp_input_image.data[tile_offset + pixel_offset + 1];
                omp_mosaic_sum[tile_index + 2] += omp_input_image.data[tile_offset + pixel_offset + 2];
            }
        }
    }

    
#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_tile_sum(&omp_input_image, omp_mosaic_sum);
#endif
}
void openmp_stage2(unsigned char* output_global_average) {
    // Calculate the average of each tile, and sum these to produce a whole image average.
    // Optionally during development call the skip function with the correct inputs to skip this stage
    // skip_compact_mosaic(TILES_X, TILES_Y, mosaic_sum, compact_mosaic, global_pixel_average);
    const int totalTiles = omp_TILES_X * omp_TILES_Y;

    int t;

    int sum_r = 0;
    int sum_g = 0;
    int sum_b = 0;

#pragma omp parallel for private(t) reduction(+: sum_r, sum_g, sum_b) 
    for (t = 0; t < totalTiles; ++t) {
        const unsigned int tile_index = t * omp_input_image.channels;

        omp_mosaic_value[tile_index] = (unsigned char)(omp_mosaic_sum[tile_index] / TILE_PIXELS);  // Integer division is fine here
        omp_mosaic_value[tile_index + 1] = (unsigned char)(omp_mosaic_sum[tile_index + 1] / TILE_PIXELS);  
        omp_mosaic_value[tile_index + 2] = (unsigned char)(omp_mosaic_sum[tile_index + 2] / TILE_PIXELS);

        sum_r += omp_mosaic_value[tile_index];
        sum_g += omp_mosaic_value[tile_index + 1];
        sum_b += omp_mosaic_value[tile_index + 2];
    }

    // divide by tile size to get average and reecombine to output_global_average
    output_global_average[0] = (unsigned char)(sum_r / (totalTiles));
    output_global_average[1] = (unsigned char)(sum_g / (totalTiles));
    output_global_average[2] = (unsigned char)(sum_b / (totalTiles));

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    //validate_compact_mosaic(omp_TILES_X, omp_TILES_Y, omp_mosaic_sum, omp_mosaic_value, output_global_average);
#endif    
}
void openmp_stage3() {
    // Broadcast the compact mosaic pixels back out to the full image size
    // Optionally during development call the skip function with the correct inputs to skip this stage
    //skip_broadcast(&omp_input_image, omp_mosaic_value, &omp_output_image);

    int t, p_x, p_y;

#pragma omp parallel for private(t, p_x, p_y)
    // For each tile
    for (t = 0; t < omp_TILES_X * omp_TILES_Y; ++t) {
        // get original loop indecies to map values to output
        const int t_x = t % omp_TILES_X;
        const int t_y = t / omp_TILES_X;

        const unsigned int tile_index = t * omp_input_image.channels;
        const unsigned int tile_offset = (t_y * omp_TILES_X * TILE_SIZE * TILE_SIZE + t_x * TILE_SIZE) * omp_input_image.channels;

        // For each pixel within the tile
        for (p_x = 0; p_x < TILE_SIZE; ++p_x) {
            for (p_y = 0; p_y < TILE_SIZE; ++p_y) {
                const unsigned int pixel_offset = (p_y * omp_input_image.width + p_x) * omp_input_image.channels;
                // Copy whole pixel
                omp_output_image.data[tile_offset + pixel_offset] = omp_mosaic_value[tile_index];
                omp_output_image.data[tile_offset + pixel_offset + 1] = omp_mosaic_value[tile_index + 1];
                omp_output_image.data[tile_offset + pixel_offset + 2] = omp_mosaic_value[tile_index + 2];

            }
        }
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // validate_broadcast(&input_image, mosaic_value, &output_image);
#endif    
}
void openmp_end(Image *output_image) {
    // Store return value
    output_image->width = omp_output_image.width;
    output_image->height = omp_output_image.height;
    output_image->channels = omp_output_image.channels;
    memcpy(output_image->data, omp_output_image.data, output_image->width * output_image->height * output_image->channels * sizeof(unsigned char));
    // Release allocations
    free(omp_output_image.data);
    free(omp_input_image.data);
    free(omp_mosaic_sum);
    free(omp_mosaic_value);
}