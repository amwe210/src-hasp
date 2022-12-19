#ifndef TENSOR_UTIL_H
#define TENSOR_UTIL_H

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <iterator>
#include <stdint.h>

#define add8_a4_a2_a3 asm volatile (".word 0x48d60777")
#define smaqa_a4_a2_a3 asm volatile (".word 0xc8d60777")
#define umaqa_a4_a2_a3 asm volatile (".word 0xccd60777")
#define smaqa_su_a4_a2_a3 asm volatile (".word 0xcad60777")

unsigned long buffer_addr = (3*1024*1024*1024) + (1024);

// #define LSB 0x000000ff
// #define PACK1x4(a) ((a & LSB) << 24 | (a & LSB) << 16 | (a & LSB) << 8 | (a & LSB))
#define PACK1x4(a) (a << 24 | a << 16 | a << 8 | a )

#define CYCLE(c) asm volatile ("rdcycle %0" : "=r" (c))

#define Offset4D(s, i0, i1, i2, i3) i0*s[0]+i1*s[1]+i2*s[2]+i3*s[3]

unsigned long long read_cycles(void)
{
  unsigned long cycles;
  asm volatile ("rdcycle %0" : "=r" (cycles));
  return (unsigned long long)cycles;
}

int ComputeBalancedPadding (int strides, int dilation_rate, int in_size, int filter_size, int out_size)
{
    int effective_filter_size = (filter_size - 1) * dilation_rate + 1;
    int total_padding =  ((out_size - 1) * strides + effective_filter_size - in_size);
    total_padding = total_padding > 0 ? total_padding : 0;
    return total_padding / 2;
}

int size4D(int *shape)
{
	return shape[0]*shape[1]*shape[2]*shape[3];
}

// inline int Offset4D(int *shape, int *strides, int i0, int i1, int i2, int i3, const char *location)
// {
// 	int dim;
// 	dim = ((i0 == 0 && shape[0] == 0) || (i0 >= 0 && i0 < shape[0])) ? 4 : 0;
// 	dim = ((i1 == 0 && shape[1] == 0) || (i1 >= 0 && i1 < shape[1])) ? 4 : 1;
// 	dim = ((i2 == 0 && shape[2] == 0) || (i2 >= 0 && i2 < shape[2])) ? 4 : 2;
// 	dim = ((i3 == 0 && shape[3] == 0) || (i3 >= 0 && i3 < shape[3])) ? 4 : 3;
// 	if(dim<4){
// 		std::cout<<"offset dim "<<dim<<" out of bounds in"<<location<<std::endl;
// 		exit(0);
// 	}
// 	// std::cout<<i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3]<<std::endl;
// 	return i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + i3 * strides[3];
// }

int *strides_from_shape_4D(int *shape)
{
	int *strides = (int*)malloc(4*sizeof(int));
	strides[3] = 1;
	strides[2] = shape[3];
	strides[1] = shape[3]*shape[2];
	strides[0] = shape[3]*shape[2]*shape[1];
	return strides;
}

// void strides_from_shape(int *shape, int *strides, int dim)
// {
// 	strides = (int*)malloc(dim*sizeof(int));
// 	int acc = 1;
// 	for (int i=dim-1; i>=0; --i){
// 		strides[i] = acc;
// 		acc *= shape[i];
// 	}
// }

struct Tensor {
    int *shape;
    int *strides;
    float *scales;
    int *zp;
    void *tensor;
};

// template <typename T>
// struct Tensor {
//     std::vector<int> shape;
//     std::vector<int> strides;
//     std::vector<double> scales;
//     std::vector<int> zp;
//     std::vector<T> tensor;
// };

struct SpTensor {
	int8_t *val;
	uint8_t *index;
	uint8_t *nnz; /*total number of non-zero vals in each partition*/
	int partitions_per_filter;
	int partition_size;
    int *shape;
    int *strides;
	float *scales;
};

void SpTfree(SpTensor *spt)
{
	free(spt->val);
	free(spt->index);
	free(spt->nnz);
}

// template <typename T>
// struct SpTensor {
// 	std::vector<T> val;
// 	std::vector<T> index;
// 	std::vector<T> nnz; /*total number of non-zero vals in each partition*/
// 	int partitions_per_filter;
// 	uint8_t partition_size;
//     std::vector<int> shape;
//     std::vector<int> strides;
// 	std::vector<double> scales;
// };

Tensor *tensors_from_file(const char *filename)
{
    std::ifstream file(filename);
	std::string num;
	std::getline(file, num);

	int total = std::stoi(num);
	Tensor *nodes = (Tensor*)malloc(total*sizeof(Tensor));
	int i = 0;
    std::string dtype, shape, scale, zp, tens;

    while (std::getline(file, dtype))
    {
		int size;
        std::getline(file, shape);
        std::istringstream shss(shape);
        std::getline(file, scale);
        std::istringstream scss(scale);
		std::getline(file, zp);
        std::istringstream zpss(zp);
		std::getline(file, tens);
        std::istringstream tss(tens);

        shss >> size;
		nodes[i].shape = (int*)malloc(size*sizeof(int));
		for (int j=0; j<size; ++j){
			shss >> nodes[i].shape[j];
		}
		// std::cout << nodes[i].shape[0] << std::endl;
		nodes[i].strides = strides_from_shape_4D(nodes[i].shape);
		scss >> size;
		nodes[i].scales = (float*)malloc(size*sizeof(float));
		for (int j=0; j<size; ++j){
			scss >> nodes[i].scales[j];
		}
		// std::cout << nodes[i].scales[0] << std::endl;
		zpss >> size;
		nodes[i].zp = (int*)malloc(size*sizeof(int));
		for (int j=0; j<size; ++j){
			zpss >> nodes[i].zp[j];
		}
		// std::cout << nodes[i].zp[0] << std::endl;
		tss >> size;
		int32_t *ptr = (int32_t*)malloc(size*sizeof(int32_t));
		for (int j=0; j<size; ++j){
			tss >> ptr[j];
		}
		// std::cout << ptr[0] << std::endl;
		nodes[i].tensor = ptr;
		if (dtype == "int8"){
			int8_t *cast = (int8_t*)malloc(size*sizeof(int8_t));
			for (int j=0; j<size; ++j){
				cast[j] = (int8_t)ptr[j];
			}
			nodes[i].tensor = cast;
			free(ptr);
		}
		++i;
    }
	return nodes;
}

// void tensors_from_file
// (
//     const char *filename, 
//     std::vector<Tensor<int8_t>> &int8_nodes, 
//     std::vector<Tensor<int32_t>> &int32_nodes, 
//     std::vector<int> &node_map
// )
// {
//     int index8 = 0;
//     int index32 = 0;
    
//     std::ifstream file(filename);
//     std::string dtype, shape_line, scale_line, zp_line, tens_line;

//     while (std::getline(file, dtype))
//     {
//         std::getline(file, shape_line);
//         std::istringstream shape_ss(shape_line);
//         std::vector<int> shape_tmp(std::istream_iterator<int>(shape_ss), {});

//         std::getline(file, scale_line);
//         std::istringstream scale_ss(scale_line);
//         std::vector<double> scale_tmp(std::istream_iterator<double>(scale_ss), {});

//         std::getline(file, zp_line);
//         std::istringstream zp_ss(zp_line);
//         std::vector<int> zp_tmp(std::istream_iterator<int>(zp_ss), {});

//         std::getline(file, tens_line);
//         std::istringstream tens_ss(tens_line);
// 		std::vector<int32_t> tens_tmp(std::istream_iterator<int32_t>(tens_ss), {});

//         if (dtype == "int8") {
//             Tensor<int8_t> node;

//             node.shape = shape_tmp;
// 			strides_from_shape(node.shape, node.strides);
//             node.scales = scale_tmp;
//             node.zp = zp_tmp;
//             std::vector<int8_t> cast(tens_tmp.begin(), tens_tmp.end());
//             node.tensor = cast;

//             int8_nodes.push_back(node);
//             node_map.push_back(index8++);
//         }
//         else if (dtype == "int32") {
//             Tensor<int32_t> node;

//             node.shape = shape_tmp;
// 			strides_from_shape(node.shape, node.strides);
//             node.scales = scale_tmp;
//             node.zp = zp_tmp;
//             node.tensor = tens_tmp;

//             int32_nodes.push_back(node);
//             node_map.push_back(index32++);
//         }
//         else {
//             std::cout<<"bad dtype in parameters file"<<std::endl;
//             exit(0);
//         }
//     }
// }

template <typename T>
T *copy(std::vector<T> &v)
{
	T *ptr = (T*)malloc(v.size()*sizeof(T));
	for (int i=0; i<v.size(); ++i){
		ptr[i] = v[i];
	}
	return ptr;
}

void compress
(
	Tensor *dense,
	SpTensor *sparse,
	int partition_size,
	bool make_NCHW /*true if compressing to NCHW format, otherwise keep NHWC*/
)
{
	int8_t *dense_tensor = (int8_t*)dense->tensor;
	sparse->shape = dense->shape;
	sparse->strides = dense->strides;
	sparse->scales = dense->scales;


	// std::cout<<sparse->strides[0]<<" "<<partition_size<<std::endl;

	// if (sparse->strides[0] % partition_size){
	// 	std::cout<<"in compress_weights partitions must be equal size"<<std::endl;
	// 	exit(0);
	// }
	
	sparse->partition_size = partition_size;
	sparse->partitions_per_filter = sparse->strides[0]/partition_size;

	if (make_NCHW){
		// int *new_shape = (int*)malloc(4*sizeof(int)); 
		// new_shape[0] = sparse->shape[0];
		// new_shape[1] = sparse->shape[3];
		// new_shape[2] = sparse->shape[1];
		// new_shape[3] = sparse->shape[2];

		// int *new_strides = (int*)malloc(4*sizeof(int));
		// new_strides[0] = sparse->strides[0];
		// new_strides[1] = sparse->strides[3];
		// new_strides[2] = sparse->strides[1];
		// new_strides[3] = sparse->strides[2];

		// sparse->shape = new_shape;
		// sparse->strides = new_strides;

		int x = sparse->shape[1];
		sparse->shape[1] = sparse->shape[3];
		sparse->shape[3] = sparse->shape[2];
		sparse->shape[2] = x;

		x = sparse->shape[1];
		sparse->shape[1] = sparse->shape[3];
		sparse->shape[3] = sparse->shape[2];
		sparse->shape[2] = x;
	}

	int count = 0;
	int pos = 0;
	std::vector<int8_t> val_tmp;
	std::vector<uint8_t> index_tmp;
	std::vector<uint8_t> nnz_tmp;

	for (int u=0; u<sparse->shape[0]; u++){
		for (int i=0; i<sparse->shape[1]; i++){
			for (int j=0; j<sparse->shape[2]; j++){
				for (int k=0; k<sparse->shape[3]; k++){
					// std::cout<<pos<<" ";
					if (pos == partition_size){
						nnz_tmp.push_back(static_cast<uint8_t>(count));
						count = 0;
						pos = 0;
					}

					// if (int8_t x = dense_tensor[Offset4D(sparse->shape, sparse->strides, u, i, j, k, "compress_weights")]){
					if (int8_t x = dense_tensor[Offset4D(sparse->strides, u, i, j, k)]){
					// if (int8_t x = dense_tensor[u*sparse->strides[0] + i*sparse->strides[1] + j*sparse->strides[2] + k*sparse->strides[3]]){
						val_tmp.push_back(x);
						index_tmp.push_back(static_cast<uint8_t>(pos));
						++count;
					}
					++pos;
				}
			}
		}
		
	}
	nnz_tmp.push_back(static_cast<uint8_t>(count));
	sparse->val = copy(val_tmp);
	sparse->index = copy(index_tmp);
	sparse->nnz = copy(nnz_tmp);
	// std::cout<<"\n";
	// for (int i=0; i<nnz_tmp.size(); ++i){
	// 	printf("nnz tmp: %d sparse->nnz: %d\n", nnz_tmp[i], sparse->nnz[i]);
	// }
}

// template <typename T>
// void compress
// (
// 	Tensor<T> &dense,
// 	SpTensor<T> &sparse,
// 	int partition_size,
// 	bool make_NCHW /*true if compressing to NCHW format, otherwise keep NHWC*/
//  )
// {
// 	sparse.shape = dense.shape;
// 	sparse.strides = dense.strides;
// 	sparse.scales = dense.scales;


// 	// std::cout<<sparse.strides[0]<<" "<<partition_size<<std::endl;

// 	// if (sparse.strides[0] % partition_size){
// 	// 	std::cout<<"in compress_weights partitions must be equal size"<<std::endl;
// 	// 	exit(0);
// 	// }
	
// 	sparse.partition_size = partition_size;
// 	sparse.partitions_per_filter = sparse.strides[0]/partition_size;

// 	if (make_NCHW){
// 		std::vector<int> new_shape = {sparse.shape[0], sparse.shape[3], sparse.shape[1], sparse.shape[2]};
// 		std::vector<int> new_strides = {sparse.strides[0], sparse.strides[3], sparse.strides[1], sparse.strides[2]};
// 		sparse.shape = new_shape;
// 		sparse.strides = new_strides;
// 	}

// 	T count = 0;
// 	T pos = 0;

// 	for (int u=0; u<sparse.shape[0]; u++){
// 		for (int i=0; i<sparse.shape[1]; i++){
// 			for (int j=0; j<sparse.shape[2]; j++){
// 				for (int k=0; k<sparse.shape[3]; k++){
// 					if (pos == partition_size){
// 						sparse.nnz.push_back(count);
// 						count = 0;
// 						pos = 0;
// 					}

// 					if (T x = dense.tensor[Offset4D(sparse.shape, sparse.strides, u, i, j, k, "compress_weights")]){
// 						sparse.val.push_back(x);
// 						sparse.index.push_back(pos);
// 						++count;
// 					}
// 					++pos;
// 				}
// 			}
// 		}
// 	}
// 	sparse.nnz.push_back(count);
// }

#endif