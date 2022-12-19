#ifndef QUANT_UTIL_H
#define QUANT_UTIL_H

#include <stdint.h>
#include <iostream>
#include <cmath>
#include <limits>

std::int32_t MultiplyByQuantizedMultiplier(std::int32_t x,
                                           std::int32_t quantized_multiplier,
                                           int shift) {
    if (shift < -31){
        std::cout<<"in MultiplyByQuantizedMultiplier, shift < -31"<<std::endl;
        exit(0);
    }

    int total_shift = 31 - shift;

    std::int64_t x_64(x);
    std::int64_t quantized_multiplier_64(quantized_multiplier);
    std::int64_t round = (int64_t)1 << (total_shift - 1);
    int64_t result = x_64 * quantized_multiplier_64 + round;
    result = result >> total_shift;

    if (result < std::numeric_limits<std::int32_t>::lowest() || result > std::numeric_limits<std::int32_t>::max())
    {
        std::cout<<"in MultiplyByQuantizedMultiplier, result out of bounds"<<std::endl;
        exit(0);
    }

    return static_cast<std::int32_t>(result);
}

// inline int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b)
// {
//   bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
//   std::int64_t a_64(a);
//   std::int64_t b_64(b);
//   std::int64_t ab_64 = a_64 * b_64;
//   std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
//   std::int32_t ab_x2_high32 =
//       static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
//   return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
// }

// template <typename IntegerType, typename ExponentType>
// inline IntegerType RoundingDivideByPOT(IntegerType x, ExponentType exponent)
// {
//     // assert(exponent >= 0);
//     // assert(exponent <= 31);
//     if (!(exponent >= 0) || !(exponent <= 31)){
//         std::cout<<"bad exponent in RoundingDivideByPOT\n";
//         exit(0);
//     }
//     // const IntegerType mask = Dup<IntegerType>((1ll << exponent) - 1);
//     const IntegerType mask = (1ll << exponent) - 1;
//     // const IntegerType zero = Dup<IntegerType>(0);
//     const IntegerType zero = 0;
//     // const IntegerType one = Dup<IntegerType>(1);
//     const IntegerType one = 1;
//     // const IntegerType remainder = BitAnd(x, mask);
//     const IntegerType remainder = x & mask; 
//     // const IntegerType threshold = Add(ShiftRight(mask, 1), BitAnd(MaskIfLessThan(x, zero), one));
//     const IntegerType threshold = (mask >> 1) + (x < 0 ? ~zero : zero) & one;
//     // return Add(ShiftRight(x, exponent), BitAnd(MaskIfGreaterThan(remainder, threshold), one));
//     return (x >> exponent) + ((remainder > threshold ? ~zero : zero) & one);
// }

// inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift)
// {
//     // using gemmlowp::RoundingDivideByPOT;
//     // using gemmlowp::SaturatingRoundingDoublingHighMul;
//     int left_shift = shift > 0 ? shift : 0;
//     int right_shift = shift > 0 ? 0 : -shift;
//     return RoundingDivideByPOT(
//         SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), 
//         right_shift);
// }

// inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift)
// {
//     // std::cout<<"in mult by quant mult"<<std::endl;
//     if (quantized_multiplier < 0){
//         std::cout<<"in MultiplyByQuantizedMultiplier, quantized_multiplier must be > 0"<<std::endl;
//         exit(0);
//     }
//     if (shift < -31 || shift > 30){
//         std::cout<<"in MultiplyByQuantizedMultiplier, shift out of bounds"<<std::endl;
//         exit(0);
//     }

//     const int64_t total_shift = 31 - shift;
//     // std::cout<<"mult by quant mult total_shift = "<<total_shift<<std::endl;
//     const int64_t round = static_cast<int64_t>(1) << (total_shift - 1);
//     // printf("mult by quant mult round = %#x\n",round);
//     int64_t result = x * static_cast<int64_t>(quantized_multiplier) + round;
//     // printf("mult by quant mult result = %#x\n",result);
//     result = result >> total_shift;
//     // printf("mult by quant mult after shift = %d\n",result);

//     if (result < std::numeric_limits<int32_t>::min() ||
//         result > std::numeric_limits<int32_t>::max())
//     {
//         std::cout<<"in MultiplyByQuantizedMultiplier, result out of bounds"<<std::endl;
//         exit(0);
//     }

//     return static_cast<int32_t>(result);
// }

enum FusedActivationType {None, Relu, Relu6, ReluN1To1};

float round(float num)
{
    if (num > 0)
        return std::floor(num + 0.5f);
    else
        return std::ceil(num - 0.5f);
}

inline void Quantize(float scale, int32_t zero_point, float f, int32_t& q) 
{
	const float tmp = round(f / scale);
	if (tmp >= static_cast<float>(std::numeric_limits<int32_t>::min()) && tmp <= static_cast<float>(std::numeric_limits<int32_t>::max())){
        q = zero_point + static_cast<int32_t>(tmp);
    }
	else {
        std::cout<<"value out of bounds in Quantize func"<<std::endl;
        exit(0);
    }
}

template <typename T>
void CalculateActivationRangeQuantized(FusedActivationType activation, Tensor *output, int32_t *act_min, int32_t *act_max) 
{
    int32_t qmin = std::numeric_limits<T>::min();
    int32_t qmax = std::numeric_limits<T>::max();
    
    const auto scale = output->scales[0];
    const auto zero_point = output->zp[0];

    int32_t tmp_q;
    if (activation == Relu) {
        Quantize(scale, zero_point, 0.0, tmp_q);
        *act_min = std::max(qmin, tmp_q);
        *act_max = qmax;
    } 
    else if (activation == Relu6) {
        Quantize(scale, zero_point, 0.0, tmp_q);
        *act_min = std::max(qmin, tmp_q);
        Quantize(scale, zero_point, 6.0, tmp_q);
        *act_max = std::min(qmax, tmp_q);
    } 
    else if (activation == ReluN1To1) {
        Quantize(scale, zero_point, -1.0, tmp_q);
        *act_min = std::max(qmin, tmp_q);
        Quantize(scale, zero_point, 1.0, tmp_q);
        *act_max = std::min(qmax, tmp_q);
    } 
    else {
        *act_min = qmin;
        *act_max = qmax;
    }
}

// template <typename T>
// void CalculateActivationRangeQuantized(FusedActivationType activation, Tensor<T> &output, int32_t *act_min, int32_t *act_max) 
// {
//     int32_t qmin = std::numeric_limits<T>::min();
//     int32_t qmax = std::numeric_limits<T>::max();
    
//     const auto scale = output.scales[0];
//     const auto zero_point = output.zp[0];

//     int32_t tmp_q;
//     if (activation == Relu) {
//         Quantize(scale, zero_point, 0.0, tmp_q);
//         *act_min = std::max(qmin, tmp_q);
//         *act_max = qmax;
//     } 
//     else if (activation == Relu6) {
//         Quantize(scale, zero_point, 0.0, tmp_q);
//         *act_min = std::max(qmin, tmp_q);
//         Quantize(scale, zero_point, 6.0, tmp_q);
//         *act_max = std::min(qmax, tmp_q);
//     } 
//     else if (activation == ReluN1To1) {
//         Quantize(scale, zero_point, -1.0, tmp_q);
//         *act_min = std::max(qmin, tmp_q);
//         Quantize(scale, zero_point, 1.0, tmp_q);
//         *act_max = std::min(qmax, tmp_q);
//     } 
//     else {
//         *act_min = qmin;
//         *act_max = qmax;
//     }
// }

double Multipler(const double input_scale, const double output_scale, const double filter_scale)
{
    return (input_scale * filter_scale) / output_scale;
}

void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift) 
{
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    const double q = std::frexp(double_multiplier, shift);
    auto q_fixed = static_cast<int64_t>(round(q * (1LL << 31)));
    if (q_fixed > (1LL << 31)){
        std::cout<<"in QuantizedMultiplier, multiplier out of bounds"<<std::endl;
        exit(0);
    }
    if (q_fixed == (1LL << 31)) {
        q_fixed /= 2;
        ++*shift;
    }
    if (q_fixed > std::numeric_limits<int32_t>::max()){
        std::cout<<"in QuantizedMultiplier, multiplier out of bounds for int32_t"<<std::endl;
        exit(0);
    }
    if (*shift < -31) {
        *shift = 0;
        q_fixed = 0;
    }
    // if (*shift > 30) {
    //     *shift = 30;
    //     q_fixed = (1LL << 31) - 1;
    }
    *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

#endif

