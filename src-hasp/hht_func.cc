#include <stdint.h>
#include <stdlib.h>

const int filter_strides_0 = 1;
const int filter_strides_1 = 1;
const int filter_strides_2 = 1;
const int filter_strides_3 = 1;

const int input_strides_0 = 1;
const int input_strides_1 = 1;
const int input_strides_2 = 1;
const int input_strides_3 = 1;

const int batches =	1;
const int input_depth = 1;
const int output_depth = 1;
const int input_height = 1;
const int input_width = 1;
const int filter_height = 1;
const int filter_width = 1;
const int output_height = 1;
const int output_width = 1;
const int stride_height = 1;
const int pad_height = 1;
const int stride_width = 1;
const int pad_width = 1;
const int dilation_height_factor = 1;
const int dilation_width_factor = 1;
const int partitions_per_filter = 1;
const int partition_size = 1;


const int buffer_size = 1;

int *nnz = (int*)malloc(partitions_per_filter*output_depth*sizeof(int));
int *index = (int*)malloc(1*sizeof(int));
int8_t *input = (int8_t*)malloc(batches*input_height*input_width*input_depth*sizeof(int8_t));
int8_t *buffer = (int8_t*)malloc(buffer_size*sizeof(int8_t));

int main ()
{
	for (int batch = 0; batch < batches; ++batch) {
        for (int x=0; x<output_height; ++x){
            for (int y = 0; y<output_width; ++y){
                int partition_base_index=0;
                for (int u=0; u<output_depth; ++u){
                    for (int p=0; p<partitions_per_filter; ++p){
                        int v_end = nnz[u*partitions_per_filter+p];
                        for (int v = partition_base_index; v<v_end; ++v) {
							int convert_index = p*partition_size + index[v];
							int filter_i = (convert_index / filter_strides_1) % filter_height;
							int filter_j = (convert_index / filter_strides_2) % filter_width;
							int filter_k = (convert_index / filter_strides_3) % input_depth;
							const int i = ((x * stride_height) - pad_height) + dilation_height_factor * filter_i;
							const int j = ((y * stride_width) - pad_width) + dilation_width_factor * filter_j;
							const bool inside_img = (i >= 0) && (i < input_height) && (j >= 0) && (j < input_width);
							if (inside_img) {
								buffer[v] = input[batch * input_strides_0 + i * input_strides_1 + j * input_strides_2 + filter_k * input_strides_3];
							}
						}
						int rem = v_end;
						while (rem < buffer_size)
							buffer[rem++] = 0;
						partition_base_index += v_end;
					}
				}
			}
		}
	}
	return 0;
}
