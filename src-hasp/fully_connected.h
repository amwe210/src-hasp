#ifndef FC_H
#define FC_H

#include "tensor_util.h"
#include "quantization_util.h"

inline void FullyConnectedDenseScalar
( 
    FusedActivationType activation, 
    Tensor *input,
    Tensor *filter,
    Tensor *bias,
    Tensor *output
)
{
    const int32_t input_offset = -input->zp[0];
    const int32_t filter_offset = -filter->zp[0];
    const int32_t output_offset = output->zp[0];
    int32_t output_multiplier;
    int output_shift;
    double effective_output_scale = static_cast<double>(input->scales[0]) * filter->scales[0] / static_cast<double>(output->scales[0]);
    QuantizeMultiplier(effective_output_scale, &output_multiplier, &output_shift);

    int32_t output_activation_min;
    int32_t output_activation_max;
    CalculateActivationRangeQuantized<int8_t>(activation, output, &output_activation_min, &output_activation_max);

    const int batches = output->shape[0];
    const int output_depth = output->shape[1];
    const int accum_depth = filter->shape[1];

    int8_t *input_tensor = (int8_t*)input->tensor;
    int8_t *filter_tensor = (int8_t*)filter->tensor;
    int32_t *bias_tensor = (int32_t*)bias->tensor;
    int8_t *output_tensor = (int8_t*)output->tensor;
  
    for (int b = 0; b < batches; ++b) {
        for (int out_c = 0; out_c < output_depth; ++out_c) {
        int32_t acc = 0;
        for (int d = 0; d < accum_depth; ++d) {
            int32_t input_val = input_tensor[b * accum_depth + d];
            int32_t filter_val = filter_tensor[out_c * accum_depth + d];
            acc += (filter_val + filter_offset) * (input_val + input_offset);
        }
        acc += bias_tensor[out_c];
        acc = MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
        acc += output_offset;
        acc = std::max(acc, output_activation_min);
        acc = std::min(acc, output_activation_max);
        output_tensor[out_c + output_depth * b] = static_cast<int8_t>(acc);
        }
    }
}




#endif