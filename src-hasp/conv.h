#ifndef CONV_H
#define CONV_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <iostream>

#include "tensor_util.h"
#include "quantization_util.h"

void PopulateConvolutionQuantizationParams
(
    Tensor *input, 
    Tensor *output, 
    const float *filter_scales,
    int32_t *per_channel_multiplier, 
    int32_t *per_channel_shift
) 
{
    // std::cout<<"in populate"<<std::endl;
    const double input_scale = input->scales[0];
    const double output_scale = output->scales[0];
    const int num_channels = output->shape[3];

    for (int i = 0; i < num_channels; ++i) {
        const double effective_output_scale = input_scale * filter_scales[i] / output_scale;
        int32_t significand;
        int channel_shift;
        QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
        per_channel_multiplier[i] = significand;
        per_channel_shift[i] = channel_shift;
        // std::cout<<significand<<" "<<channel_shift<<std::endl;
    }
}

void ScalarDenseConv
(
    int dilation_width_factor, 
    int dilation_height_factor,
    int stride_width, 
    int stride_height,  
    FusedActivationType activation, 
    Tensor *input,
    Tensor *filter,
    Tensor *bias,
    Tensor *output
)
{
    const int quantized_channels = output->shape[3];
    int32_t *output_multiplier = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    int32_t *output_shift = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    PopulateConvolutionQuantizationParams(input, output, filter->scales, output_multiplier, output_shift);
	// int8_t input_offset = static_cast<int8_t>(-input->zp[0]);
    int32_t input_offset = -input->zp[0];
	// int8_t output_offset = static_cast<int8_t>(output->zp[0]);
    int32_t output_offset = output->zp[0];
    // std::cout<<"done with quant params"<<std::endl;

    int32_t output_activation_min; 
    int32_t output_activation_max; 
    CalculateActivationRangeQuantized<int8_t>(activation, output, &output_activation_min, &output_activation_max);
    // std::cout<<"done with quant range"<<std::endl;

    const int batches =	input->shape[0];
    const int input_depth = input->shape[3];
    const int output_depth = output->shape[3];

    const int input_height = input->shape[1];
    const int input_width = input->shape[2];
    const int filter_height = filter->shape[1];
    const int filter_width = filter->shape[2];
    const int output_height = output->shape[1];
    const int output_width = output->shape[2];
    // std::cout<<"done with var assign"<<std::endl;

    int pad_height = ComputeBalancedPadding(stride_height, dilation_height_factor, input_height, filter_height, output_height);
    int pad_width = ComputeBalancedPadding(stride_height, dilation_height_factor, input_width, filter_width, output_width);
    // std::cout<<"done with padding"<<std::endl;

    int8_t *input_tensor = (int8_t*)input->tensor;
    int8_t *filter_tensor = (int8_t*)filter->tensor;
    int32_t *bias_tensor = (int32_t*)bias->tensor;
    int8_t *output_tensor = (int8_t*)output->tensor;

    int input_stride_0 = input->strides[0];
    int input_stride_1 = input->strides[1];
    int input_stride_2 = input->strides[2];
    int input_stride_3 = input->strides[3];

    int filter_stride_0 = filter->strides[0];
    int filter_stride_1 = filter->strides[1];
    int filter_stride_2 = filter->strides[2];
    int filter_stride_3 = filter->strides[3];

    int output_stride_0 = output->strides[0];
    int output_stride_1 = output->strides[1];
    int output_stride_2 = output->strides[2];
    int output_stride_3 = output->strides[3];


    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    // std::cout<<"out channel: "<<out_channel<<std::endl;
                    int32_t acc = 0;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;
                            // Zero padding by omitting the areas outside the image.
                            const bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);
                            // std::cout<<in_y<<" "<<in_x<<std::endl;
                            // std::cout<<filter_y<<" "<<filter_x<<std::endl;

                            if (is_point_inside_image) {
                                for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                                    int32_t input_val = input_tensor[batch*input_stride_0 + in_y*input_stride_1 +  in_x*input_stride_2 + in_channel*input_stride_3];
                                    // printf("input val: %d = %d + %d\n", input_val+input_offset, input_val, input_offset);
                                    // printf("input val: %#x\n", static_cast<uint8_t>(input_val+input_offset));
                                    int32_t filter_val = filter_tensor[out_channel*filter_stride_0 + filter_y*filter_stride_1 + filter_x*filter_stride_2 + in_channel*filter_stride_3];
                                    // /*if (filter_val > 0)*/ printf("input val: %d\n", input_val);
                                    // printf("filter val: %#x\n", filter_val);
                                    // printf("[%d][%d][%d][%d]: %#x\n", out_channel, filter_y, filter_x, in_channel, filter_val);
                                    acc += filter_val * (input_val + input_offset);
                                    // printf("after in[%d][%d][%d] * w[%d][%d][%d]: %d\n", in_y, in_x, in_channel, filter_y, filter_x, in_channel, acc);
                                }
                            }
                        }
                    }
                    // std::cout<<acc<<" \n";
                    acc += bias_tensor[out_channel];
                    // std::cout<<acc<<" ";
                    acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
                    // std::cout<<acc<<" ";
                    acc += output_offset;
                    // std::cout<<acc<<" ";
                    acc = std::max(acc, output_activation_min);
                    // std::cout<<acc<<" ";
                    acc = std::min(acc, output_activation_max);
                    // std::cout<<acc<<std::endl;
                    // output_tensor[Offset4D(output->shape, output->strides, batch, out_y, out_x, out_channel, "conv validate out")] = static_cast<int8_t>(acc);
                    output_tensor[batch*output_stride_0 + out_y*output_stride_1 + out_x*output_stride_2 + out_channel*output_stride_3] = static_cast<int8_t>(acc);
                    // std::cout<<"out["<<Offset4D(output.shape, output.strides, batch, out_y, out_x, out_channel, "conv validate out")<<"] = "<<(int)output.tensor[Offset4D(output.shape, output.strides, batch, out_y, out_x, out_channel, "conv validate out")]<<std::endl;
                }
            }
        }
    }
}

void ConvPackedQuantizedDenseCPU
(
    int dilation_width_factor, 
    int dilation_height_factor,
    int stride_width, 
    int stride_height,  
    FusedActivationType activation, 
    Tensor *input,
    Tensor *filter,
    Tensor *bias,
    Tensor *output
)
{
    const int quantized_channels = output->shape[3];
    int32_t *output_multiplier = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    int32_t *output_shift = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    PopulateConvolutionQuantizationParams(input, output, filter->scales, output_multiplier, output_shift);
	// int8_t input_offset = static_cast<int8_t>(-input->zp[0]);
	// int8_t output_offset = static_cast<int8_t>(output->zp[0]);
    int32_t input_offset = -input->zp[0];
	int32_t output_offset = output->zp[0];
    const bool require_su = input_offset == 128;
    // uint32_t packed_input_offset = PACK1x4(input_offset);

    int32_t output_activation_min; 
    int32_t output_activation_max; 
    CalculateActivationRangeQuantized<int8_t>(activation, output, &output_activation_min, &output_activation_max);

    const int batches =	input->shape[0];
    const int input_depth = input->shape[3];
    const int output_depth = output->shape[3];

    const int input_height = input->shape[1];
    const int input_width = input->shape[2];
    const int filter_height = filter->shape[1];
    const int filter_width = filter->shape[2];
    const int output_height = output->shape[1];
    const int output_width = output->shape[2];

    int pad_height = ComputeBalancedPadding(stride_height, dilation_height_factor, input_height, filter_height, output_height);
    int pad_width = ComputeBalancedPadding(stride_height, dilation_height_factor, input_width, filter_width, output_width);

    int8_t *input_tensor = (int8_t*)input->tensor;
    int8_t *filter_tensor = (int8_t*)filter->tensor;
    // uint32_t *filter_tensor = (uint32_t*)filter->tensor;
    int32_t *bias_tensor = (int32_t*)bias->tensor;
    int8_t *output_tensor = (int8_t*)output->tensor;

    int input_stride_0 = input->strides[0];
    int input_stride_1 = input->strides[1];
    int input_stride_2 = input->strides[2];
    int input_stride_3 = input->strides[3];

    int filter_stride_0 = filter->strides[0];
    int filter_stride_1 = filter->strides[1];
    int filter_stride_2 = filter->strides[2];
    int filter_stride_3 = filter->strides[3];

    int output_stride_0 = output->strides[0];
    int output_stride_1 = output->strides[1];
    int output_stride_2 = output->strides[2];
    int output_stride_3 = output->strides[3];

    for (int batch = 0; batch < batches; ++batch) {
        for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
            for (int out_x = 0; out_x < output_width; ++out_x) {
                const int in_x_origin = (out_x * stride_width) - pad_width;
                for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
                    int pack_count = 0;
                    uint32_t packed_input = 0;
                    uint32_t *packed_filter = (uint32_t*)(&filter_tensor[out_channel*filter_stride_0]);
                    // uint32_t *packed_filter = &filter_tensor[out_channel*filter_stride_0];
                    // printf("init [%d][%d][%d][%d]: filter = %#x, tensor index = %d\n", batch, out_y, out_x, out_channel, *packed_filter, out_channel*filter_stride_0);
                    //std::cout<<"out channel: "<<out_channel<<std::endl;
                    int32_t acc = 0;
                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;
                            // Zero padding by omitting the areas outside the image.
                            const bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);
                            // std::cout<<in_y<<" "<<in_x<<std::endl;
                            // std::cout<<filter_y<<" "<<filter_x<<std::endl;

                            for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                                if (is_point_inside_image) {
                                    int32_t input_val = input_tensor[batch*input_stride_0 + in_y*input_stride_1 +  in_x*input_stride_2 + in_channel*input_stride_3];
                                    // printf("input val: %d\n", input_val);
                                    input_val += input_offset;
                                    packed_input = packed_input | static_cast<uint8_t>(input_val) << pack_count*8;
                                    // printf("input val: %d\n", input_val);
                                    // printf("packed input: %#x\n", packed_input);
                                    // int fval = static_cast<int8_t>(*packed_filter >> pack_count*8);
                                    // printf("filter val: %d\n", fval);
                                    // printf("[%d][%d][%d][%d]: %#x\n", out_channel, filter_y, filter_x, in_channel, *packed_filter);
                                }

                                if (++pack_count == 4){
                                    // printf("[%d][%d][%d][%d]: filter = %#x input = %#x\n", out_channel, filter_y, filter_x, in_channel, *packed_filter, packed_input);
                                    // printf("filter: %#x\n", *packed_filter);
                                    //packed_input full, so add, multiply, and accumulate
                                    // asm("lw a2, 0(%0);" :: "r"(&packed_input) : );
                                    // asm("lw a3, 0(%0);" :: "r"(&packed_input_offset) : );
                                    // add8_a4_a2_a3;
                                    // asm("sw a4, 0(%0);" :: "r"(&packed_input) : );
                                    
                                    asm("lw a4, 0(%0);" :: "r"(&acc) : );
                                    asm("lw a2, 0(%0);" :: "r"(packed_filter) : );
                                    asm("lw a3, 0(%0);" :: "r"(&packed_input) : );
                                    // smaqa_a4_a2_a3;
                                    if (require_su) smaqa_su_a4_a2_a3;
                                    else smaqa_a4_a2_a3;
                                    asm("sw a4, 0(%0);" :: "r"(&acc) : );
                                    // printf("after in[%d][%d][%d] * w[%d][%d][%d]: %d\n", in_y, in_x, in_channel, filter_y, filter_x, in_channel, acc);

                                    pack_count = 0;
                                    packed_input = 0;
                                    ++packed_filter;
                                }
                            }

                            // if (is_point_inside_image) {
                            //     for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
                            //         int8_t input_val = input_tensor[batch*input_stride_0 + in_y*input_stride_1 +  in_x*input_stride_2 + in_channel*input_stride_3];
                            //         // printf("input val: %d\n", input_val);
                            //         packed_input = packed_input | input_val << pack_count*8;

                            //         if (++pack_count == 4){
                            //             // printf("filter val: %#x\n", packed_filter);
                            //             //packed_input full, so add, multiply, and accumulate
                            //             asm("lw a2, 0(%0);" :: "r"(&packed_input) : );
                            //             asm("lw a3, 0(%0);" :: "r"(&packed_input_offset) : );
                            //             add8_a4_a2_a3;
                            //             asm("sw a4, 0(%0);" :: "r"(&packed_input) : );
                                        
                            //             asm("lw a4, 0(%0);" :: "r"(&acc) : );
                            //             asm("lw a2, 0(%0);" :: "r"(packed_filter) : );
                            //             asm("lw a3, 0(%0);" :: "r"(&packed_input) : );
                            //             smaqa_a4_a2_a3;
                            //             asm("sw a4, 0(%0);" :: "r"(&acc) : );
                            //             // std::cout<<"acc: "<<acc<<std::endl;

                            //             pack_count = 0;
                            //             packed_input = 0;
                            //             ++packed_filter;
                            //         }
                            //     }
                            // }
                            // else{
                            //     pack_count += input_depth;
                            //     packed_filter += pack_count/4;
                            //     pack_count = pack_count%4;
                            // }
                        }
                    }
                    if (pack_count){
                        // printf("remainder [%d][%d][%d][%d]: filter = %#x input = %#x\n", batch, out_y, out_x, out_channel, *packed_filter, packed_input);
                        // printf("filter: %#x\n", *packed_filter);
                        // asm("lw a2, 0(%0);" :: "r"(&packed_input) : );
                        // asm("lw a3, 0(%0);" :: "r"(&packed_input_offset) : );
                        // add8_a4_a2_a3;
                        // asm("sw a4, 0(%0);" :: "r"(&packed_input) : );
                        
                        asm("lw a4, 0(%0);" :: "r"(&acc) : );
                        asm("lw a2, 0(%0);" :: "r"(packed_filter) : );
                        asm("lw a3, 0(%0);" :: "r"(&packed_input) : );
                        // smaqa_a4_a2_a3;
                        if (require_su) smaqa_su_a4_a2_a3;
                        else smaqa_a4_a2_a3;
                        asm("sw a4, 0(%0);" :: "r"(&acc) : );
                        // std::cout<<"rem acc: "<<acc<<std::endl;
                    }
                    // std::cout<<acc<<" ";
                    acc += bias_tensor[out_channel];
                    // std::cout<<acc<<" ";
                    acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
                    // std::cout<<acc<<" ";
                    acc += output_offset;
                    // std::cout<<acc<<" ";
                    acc = std::max(acc, output_activation_min);
                    // std::cout<<acc<<" ";
                    acc = std::min(acc, output_activation_max);
                    // std::cout<<acc<<std::endl;
                    // output_tensor[Offset4D(output.shape, output.strides, batch, out_y, out_x, out_channel, "conv validate out")] = static_cast<int8_t>(acc);
                    output_tensor[batch*output_stride_0 + out_y*output_stride_1 + out_x*output_stride_2 + out_channel*output_stride_3] = static_cast<int8_t>(acc);
                    //std::cout<<"out["<<Offset4D(output.shape, output.strides, batch, out_y, out_x, out_channel, "conv validate out")<<"] = "<<(int)output.tensor[Offset4D(output.shape, output.strides, batch, out_y, out_x, out_channel, "conv validate out")]<<std::endl;
                }
            }
        }
    }
}

void ConvPackedQuantizedSparseCPU
(
    int dilation_width_factor, 
    int dilation_height_factor,
    int stride_width, 
    int stride_height,  
    FusedActivationType activation, 
    Tensor *input,
    SpTensor *filter,
    Tensor *bias,
    Tensor *output
)
{
    const int quantized_channels = output->shape[3];
    int32_t *output_multiplier = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    int32_t *output_shift = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    PopulateConvolutionQuantizationParams(input, output, filter->scales, output_multiplier, output_shift);
	// int8_t input_offset = static_cast<int8_t>(-input->zp[0]);
	// int8_t output_offset = static_cast<int8_t>(output->zp[0]);
    int32_t input_offset = -input->zp[0];
	int32_t output_offset = output->zp[0];
    const bool require_su = input_offset == 128;
    // uint32_t packed_input_offset = PACK1x4(input_offset);

    int32_t output_activation_min; 
    int32_t output_activation_max; 
    CalculateActivationRangeQuantized<int8_t>(activation, output, &output_activation_min, &output_activation_max);

    const int batches =	input->shape[0];
    const int input_depth = input->shape[3];
    const int output_depth = output->shape[3];

    const int input_height = input->shape[1];
    const int input_width = input->shape[2];
    const int filter_height = filter->shape[1];
    const int filter_width = filter->shape[2];
    const int output_height = output->shape[1];
    const int output_width = output->shape[2];

    int pad_height = ComputeBalancedPadding(stride_height, dilation_height_factor, input_height, filter_height, output_height);
    int pad_width = ComputeBalancedPadding(stride_height, dilation_height_factor, input_width, filter_width, output_width);

    int8_t *input_tensor = (int8_t*)input->tensor;
    int32_t *bias_tensor = (int32_t*)bias->tensor;
    int8_t *output_tensor = (int8_t*)output->tensor;

    int input_stride_0 = input->strides[0];
    int input_stride_1 = input->strides[1];
    int input_stride_2 = input->strides[2];
    int input_stride_3 = input->strides[3];

    int filter_stride_0 = filter->strides[0];
    int filter_stride_1 = filter->strides[1];
    int filter_stride_2 = filter->strides[2];
    int filter_stride_3 = filter->strides[3];

    int output_stride_0 = output->strides[0];
    int output_stride_1 = output->strides[1];
    int output_stride_2 = output->strides[2];
    int output_stride_3 = output->strides[3];

    // unsigned long counter = 0;

    for (int batch = 0; batch < batches; ++batch) {
        for (int x=0; x<output_height; ++x){
            // std::cout<<"begin out row: "<<x<<std::endl;
            const int i_origin = (x * stride_height) - pad_height;
            for (int y = 0; y<output_width; ++y){
                const int j_origin = (y * stride_width) - pad_width;
                int partition_base_index=0;
                for (int u=0; u<output_depth; ++u){
                    //loop through each partition, 4 values at a time
                    int32_t acc = 0;
                    for (int p=0; p<filter->partitions_per_filter; ++p){
                        // std::cout<<"index into nnz[]"<<u*filter->partitions_per_filter+p<<std::endl;
                        int v_end = filter->nnz[u*filter->partitions_per_filter+p];
                        uint32_t *packed_filter = (uint32_t*)(&filter->val[partition_base_index]);
                        // std::cout<<"v_end: "<<v_end<<std::endl;
                        for (int v = 0; v<v_end; v+=4) {
                            uint32_t packed_input = 0;
                            for (int pack_offset=0; pack_offset<4; ++pack_offset){
                                // unsigned long long s0 = read_cycles();
                                int32_t input_val = 0;
                                if (v+pack_offset < v_end){
                                    // unsigned long long s1 = read_cycles();
                                    int convert_index = p*filter->partition_size + filter->index[partition_base_index+v+pack_offset];
                                    // std::cout<<"convert_index: "<<convert_index<<std::endl;
                                    int filter_i = (convert_index / filter_stride_1) % filter_height; //val_position[0]
                                    int filter_j = (convert_index / filter_stride_2) % filter_width; //val_position[1]
                                    int filter_k = (convert_index / filter_stride_3) % input_depth; //val_position[2]
                                    const int i = i_origin + dilation_height_factor * filter_i;
                                    const int j = j_origin + dilation_width_factor * filter_j;
                                    const bool inside_img = (i >= 0) && (i < input_height) && (j >= 0) && (j < input_width);
                                    // std::cout<<"filter: i="<<filter_i<<" j="<<filter_j<<" k="<<filter_k<<std::endl;
                                    // std::cout<<"input: i="<<i<<" j="<<i<<" k="<<filter_k<<std::endl;
                                    // unsigned long long e1 = read_cycles();
                                    // printf("offset cycles %d\n",e1-s1);
                                    if (inside_img) {
                                        // unsigned long long s2 = read_cycles(); 
                                        // input_val = input_tensor[Offset4D(input->shape, input->strides, batch, i, j, filter_k, "conv input val")];
                                        input_val = input_tensor[batch*input_stride_0 + i*input_stride_1 +  j*input_stride_2 + filter_k*input_stride_3];
                                        input_val += input_offset;
                                        // printf("input val: %#x\n", input_val);
                                        // unsigned long long e2 = read_cycles();
                                        // printf("input val cycles %d\n",e2-s2);
                                    }
                                    // else
                                    //     input_val = 0;
                                    //printf("SparseCPU: input[%d][%d][%d]: %d\n", i, j, filter_k, input_val);
                                }
                                // else 
                                //     input_val = 0;
                                // std::cout<<"packed offset: "<<pack_offset<<std::endl;
                                // std::cout<<"input val: "<<(int)input_val<<std::endl;
                                packed_input = packed_input | static_cast<uint8_t>(input_val) << pack_offset*8;
                                // packed_input = packed_input | (input_val & LSB) << pack_offset*8;
                                // std::ios_base::fmtflags f(std::cout.flags());
                                // std::cout<<"updated packed input: "<<std::hex<<packed_input<<std::endl;
                                // std::cout<<"packed input offset: "<<packed_input_offset<<std::endl;
                                // std::cout.flags(f);
                                // unsigned long long e0 = read_cycles();
                                // printf("cpu packing cycles %d: %d\n",counter, e0-s0);
                                // ++counter;
                            }
                            // printf("CPU: packed input %d for output[%d][%d][%d] = %#x\n", v, x, y, u, packed_input);
                            // add8 packed input offset
                            // std::ios_base::fmtflags f(std::cout.flags());
                            // std::cout<<"updated packed input: "<<std::hex<<packed_input<<std::endl;
                            // std::cout<<"packed input offset: "<<packed_input_offset<<std::endl;
                            // asm("lw a4, 0(%0);" :: "r"(&packed_input) : );
                            // asm("lw a2, 0(%0);" :: "r"(&packed_input) : );
                            // asm("lw a3, 0(%0);" :: "r"(&packed_input_offset) : );
                            // add8_a4_a2_a3;
                            // asm("sw a4, 0(%0);" :: "r"(&packed_input) : );

                            // std::cout<<"updated packed input: "<<std::hex<<packed_input<<std::endl;
                            // std::cout<<"acc: "<<acc<<std::endl;
                            // std::cout<<"filter.val[partition_base_index+v]: "<<*(uint32_t*)(&filter.val[partition_base_index+v])<<std::endl;
                            // std::cout<<"filter val offset: "<<(&filter.val[0]) - (&filter.val[partition_base_index+v])<<std::endl;

                            // uint32_t r2, r3, r4;

                            //SIMD packed multiply and reduce
                            // uint32_t *packed_filter = (uint32_t*)(&filter->val[partition_base_index+v]);
                            // printf("packed input: %#x\n", packed_input);
                            asm("lw a4, 0(%0);" :: "r"(&acc) : );
                            // asm("sw a4, 0(%0);" :: "r"(&r4) : );
                            asm("lw a2, 0(%0);" :: "r"(packed_filter) : );
                            // asm("sw a2, 0(%0);" :: "r"(&r2) : );
                            asm("lw a3, 0(%0);" :: "r"(&packed_input) : );
                            // asm("sw a3, 0(%0);" :: "r"(&r3) : );
                            if (require_su) smaqa_su_a4_a2_a3;
                            else smaqa_a4_a2_a3;
                            // smaqa_a4_a2_a3;
                            asm("sw a4, 0(%0);" :: "r"(&acc) : );

                            ++packed_filter;

                            // std::cout<<"packed_input: "<<packed_input<<std::endl;
                            // std::cout<<"r2, r3, r4: "<<r2<<" "<<r3<<" "<<r4<<std::endl;
                            // std::cout<<"acc: "<<acc<<std::endl;
                            // std::cout.flags(f);

                            // std::cout<<"acc: "<<acc<<std::endl;
                        }
                        partition_base_index += v_end;
                    }
                    // std::cout<<acc<<" ";
                    acc += bias_tensor[u];
                    // std::cout<<acc<<" ";
                    acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[u], output_shift[u]);
                    // std::cout<<acc<<" ";
                    acc += output_offset;
                    // std::cout<<acc<<"\n";
                    // acc = acc > output_activation_min ? acc : output_activation_min;
                    // std::cout<<acc<<" ";
                    // acc = acc < output_activation_max ? acc : output_activation_max;
                    acc = std::max(acc, output_activation_min);
                    // std::cout<<acc<<" ";
                    acc = std::min(acc, output_activation_max);
                    // std::cout<<acc<<std::endl;
                    // output_tensor[Offset4D(output->shape, output->strides, batch, x, y, u, "conv output")] = static_cast<int8_t>(acc);
                    output_tensor[batch*output_stride_0 + x*output_stride_1 + y*output_stride_2 + u*output_stride_3] = static_cast<int8_t>(acc);
                    // std::cout<<"out["<<Offset4D(output->shape, output->strides, batch, x, y, u, "spconv output")<<"] = "<<(int)output_tensor[Offset4D(output->shape, output->strides, batch, x, y, u, "conv output")]<<std::endl;
                }
            }
        }
    }
}


void ConvPackedQuantizedSparseHHT
(
    int dilation_width_factor, 
    int dilation_height_factor,
    int stride_width, 
    int stride_height,  
    FusedActivationType activation, 
    Tensor *input,
    SpTensor *filter,
    Tensor *bias,
    Tensor *output
)
{
    const int quantized_channels = output->shape[3];
    int32_t *output_multiplier = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    int32_t *output_shift = (int32_t*)malloc(quantized_channels*sizeof(int32_t));
    PopulateConvolutionQuantizationParams(input, output, filter->scales, output_multiplier, output_shift);
	// int8_t input_offset = static_cast<int8_t>(-input->zp[0]);
	// int8_t output_offset = static_cast<int8_t>(output->zp[0]);
    int32_t input_offset = -input->zp[0];
	int32_t output_offset = output->zp[0];
    const bool require_su = input_offset == 128;
    // uint32_t packed_input_offset = PACK1x4(input_offset);

    int32_t output_activation_min; 
    int32_t output_activation_max; 
    CalculateActivationRangeQuantized<int8_t>(activation, output, &output_activation_min, &output_activation_max);

    const int batches =	input->shape[0];
    const int input_depth = input->shape[3];
    const int output_depth = output->shape[3];

    const int input_height = input->shape[1];
    const int input_width = input->shape[2];
    const int filter_height = filter->shape[1];
    const int filter_width = filter->shape[2];
    const int output_height = output->shape[1];
    const int output_width = output->shape[2];

    int pad_height = ComputeBalancedPadding(stride_height, dilation_height_factor, input_height, filter_height, output_height);
    int pad_width = ComputeBalancedPadding(stride_height, dilation_height_factor, input_width, filter_width, output_width);

    int8_t *input_tensor = (int8_t*)input->tensor;
    int32_t *bias_tensor = (int32_t*)bias->tensor;
    int8_t *output_tensor = (int8_t*)output->tensor;

    // printf("output tensor addr: %#x, %#x\n", output_tensor, output->tensor);
    // printf("input tensor addr: %#x, %#x\n", input_tensor, input->tensor);
    // printf("partition size: %d\n", filter->partition_size);
    // std::cout<<size4D(output->shape)<<"\n";

    int input_stride_0 = input->strides[0];
    int input_stride_1 = input->strides[1];
    int input_stride_2 = input->strides[2];
    int input_stride_3 = input->strides[3];

    int filter_stride_0 = filter->strides[0];
    int filter_stride_1 = filter->strides[1];
    int filter_stride_2 = filter->strides[2];
    int filter_stride_3 = filter->strides[3];

    int output_stride_0 = output->strides[0];
    int output_stride_1 = output->strides[1];
    int output_stride_2 = output->strides[2];
    int output_stride_3 = output->strides[3];

    int *p;
    p = (3*1024*1024*1024);
    *(p) = 0;
    *(p+1) = 8; //format
    *(p+25) = input_depth;
    *(p+26) = output_depth;
    *(p+34) = input_tensor;
    *(p+35) = input_height;
    *(p+36) = input_width;
    *(p+37) = output_height;
    *(p+38) = output_width;
    *(p+39) = stride_height;
    *(p+41) = filter->partitions_per_filter;
    *(p+42) = filter->partition_size;
    *(p+43) = filter->nnz;
    *(p+44) = batches;
    *(p+45) = pad_height;
    *(p+46) = filter_stride_0;
    *(p+47) = filter_stride_1;
    *(p+48) = filter_stride_2;
    *(p+49) = filter_stride_3;
    *(p+50) = filter->index;
    *(p+51) = filter_height;
    *(p+52) = filter_width;
    *(p+53) = input_depth;
    *(p+54) = dilation_height_factor;
    *(p+55) = input_stride_0;
    *(p+56) = input_stride_1;
    *(p+57) = input_stride_2;
    *(p+58) = input_stride_3;
    *(p+59) = input_offset;

    // printf("CPU: filter array addresses: val %#x / index %#x / nnz %#x\n", &input.tensor[0], &filter.index[0], &filter.nnz[0]);

    volatile int *bc_active = p+12;
    volatile int *func_start = p+40;
    volatile int *read_done_flag = p+60;
    volatile int8_t *buffer = buffer_addr;
    // printf("HHT start\n");
    // *(bc_active) = 1;
    *(func_start) = 1;

    for (int batch = 0; batch < batches; ++batch) {
        for (int x=0; x<output_height; ++x){
            // std::cout<<"begin out row: "<<x<<std::endl;
            for (int y = 0; y<output_width; ++y){
                int partition_base_index=0;
                for (int u=0; u<output_depth; ++u){
                    int32_t acc = 0;
                    for (int p=0; p<filter->partitions_per_filter; ++p){
                        // std::cout<<"CPU: index into nnz[]"<<u*filter->partitions_per_filter+p<<std::endl;
                        uint32_t *packed_filter = (uint32_t*)(&filter->val[partition_base_index]);
                        int v_end = filter->nnz[u*filter->partitions_per_filter+p];
                        // std::cout<<"CPU: v_end: "<<v_end<<std::endl;
                        for (int v = 0; v<v_end; v+=4) {
                            // while(*(bc_active)==1) {
                            //     // printf("CPU: bc_active %d\n", *(bc_active));
                            //     // if(*(bc_active)==0) break;
                            // }                        
                            int32_t packed_input = *(uint32_t*)(&buffer[v]);
                            // int32_t packed_input = buffer[v];
                            // asm("lw a4, 0(%0);" :: "r"(&packed_input) : );
                            // asm("lw a2, 0(%0);" :: "r"(&packed_input) : );
                            // asm("lw a3, 0(%0);" :: "r"(&packed_input_offset) : );
                            // add8_a4_a2_a3;
                            // asm("sw a4, 0(%0);" :: "r"(&packed_input) : );

                            // uint32_t r2, r3, r4;

                            // uint32_t *packed_filter = (uint32_t*)(&filter->val[partition_base_index+v]);
                            // printf("CPU: packed filter %d for output[%d][%d][%d] = %#x\n", v, x, y, u, *packed_filter);
                            // printf("CPU: packed input %d for output[%d][%d][%d] = %#x\n", v, x, y, u, packed_input);

                            //SIMD packed multiply and reduce
                            asm("lw a4, 0(%0);" :: "r"(&acc) : );
                            asm("lw a2, 0(%0);" :: "r"(packed_filter) : );
                            asm("lw a3, 0(%0);" :: "r"(&packed_input) : );
                            if (require_su) smaqa_su_a4_a2_a3;
                            else smaqa_a4_a2_a3;
                            // smaqa_a4_a2_a3;
                            asm("sw a4, 0(%0);" :: "r"(&acc) : );

                            ++packed_filter;
                        }
                        // *(bc_active) = 1;
                        *(read_done_flag) = 0;
                        partition_base_index += v_end;
                    }
                    // std::cout<<acc<<" ";
                    acc += bias_tensor[u];
                    // std::cout<<acc<<"\n";
                    acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[u], output_shift[u]);
                    // std::cout<<acc<<"\n ";
                    acc += output_offset;
                    // std::cout<<acc<<"\n ";
                    // acc = acc > output_activation_min ? acc : output_activation_min;
                    // std::cout<<acc<<" ";
                    // acc = acc < output_activation_max ? acc : output_activation_max;
                    acc = std::max(acc, output_activation_min);
                    // std::cout<<acc<<" ";
                    acc = std::min(acc, output_activation_max);
                    // std::cout<<acc<<std::endl;
                    // output.tensor[Offset4D(output->shape, output->strides, batch, x, y, u, "conv output")] = static_cast<int8_t>(acc);
                    // std::cout<<"output offset: "<<batch*output_stride_0 + x*output_stride_1 + y*output_stride_2 + u*output_stride_3<<"\n";
                    // int output_offset = batch*output_stride_0 + x*output_stride_1 + y*output_stride_2 + u*output_stride_3;
                    // printf("%d\n",outcount);
                    // output_tensor[outcount++] = static_cast<int8_t>(acc);
                    output_tensor[batch*output_stride_0 + x*output_stride_1 + y*output_stride_2 + u*output_stride_3] = static_cast<int8_t>(acc);
                    // output_tensor[0] = static_cast<int8_t>(acc);
                    // std::cout<<"out["<<Offset4D(output->shape, output->strides, batch, x, y, u, "spconv output")<<"] = "<<(int)output->tensor[Offset4D(output->shape, output->strides, batch, x, y, u, "conv output")]<<std::endl;
                }
            }
        }
    }
}


#endif