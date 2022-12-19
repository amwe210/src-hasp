#ifndef POOLING_H
#define POOLING_H

#include "tensor_util.h"
#include "quantization_util.h"

inline void AveragePool
(
    int stride_height,
    int stride_width,
    int filter_height, 
    int filter_width,
    Tensor *input,
    Tensor *output
) 
{
    int32_t output_activation_min; 
    int32_t output_activation_max; 
    CalculateActivationRangeQuantized<int8_t>(None, output, &output_activation_min, &output_activation_max);

    const int batches = input->shape[0];
    const int depth = input->shape[3];
    const int input_height = input->shape[1];
    const int input_width = input->shape[2];
    const int output_height = output->shape[1];
    const int output_width = output->shape[2];

    int pad_height = ComputeBalancedPadding(stride_height, 1, input_height, filter_height, output_height);
    int pad_width = ComputeBalancedPadding(stride_height, 1, input_width, filter_width, output_width);

    int input_stride_0 = input->strides[0];
    int input_stride_1 = input->strides[1];
    int input_stride_2 = input->strides[2];
    int input_stride_3 = input->strides[3];

    int output_stride_0 = output->strides[0];
    int output_stride_1 = output->strides[1];
    int output_stride_2 = output->strides[2];
    int output_stride_3 = output->strides[3];

    int8_t *input_tensor = (int8_t*)input->tensor;
    int8_t *output_tensor = (int8_t*)output->tensor;
    
    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
            for (int out_x = 0; out_x < output_width; ++out_x) {
                for (int channel = 0; channel < depth; ++channel) {
                    const int in_x_origin = (out_x * stride_width) - pad_width;
                    const int in_y_origin = (out_y * stride_height) - pad_height;
                    // Compute the boundaries of the filter region clamped so as to
                    // ensure that the filter window fits in the input array.
                    const int filter_x_start = std::max(0, -in_x_origin);
                    const int filter_x_end = std::min(filter_width, input_width - in_x_origin);
                    const int filter_y_start = std::max(0, -in_y_origin);
                    const int filter_y_end = std::min(filter_height, input_height - in_y_origin);
                    int32_t acc = 0;
                    int filter_count = 0;
                    for (int filter_y = filter_y_start; filter_y < filter_y_end; ++filter_y) {
                        for (int filter_x = filter_x_start; filter_x < filter_x_end; ++filter_x) {
                            const int in_x = in_x_origin + filter_x;
                            const int in_y = in_y_origin + filter_y;
                            acc += input_tensor[batch*input_stride_0 + in_y*input_stride_1 + in_x*input_stride_2 + channel*input_stride_3];
                            filter_count++;
                        }
                    }
                    if (filter_count == 0) return false;
                    // Round to the closest integer value.
                    acc = acc > 0 ? (acc + filter_count / 2) / filter_count : (acc - filter_count / 2) / filter_count;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);
                    output_tensor[batch*output_stride_0 + out_y*output_stride_1 + out_x*output_stride_2 + channel*output_stride_3] = static_cast<int8_t>(acc);
                }
            }
        }
    }
}

#endif