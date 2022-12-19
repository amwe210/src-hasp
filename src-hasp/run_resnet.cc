#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <iterator>
#include <stdint.h>
#include <vector>
#include <time.h>
#include "tensor_util.h"
#include "conv.h"
#include "add.h"
#include "pooling.h"
#include "fully_connected.h"

#define NARGS 6
void usage()
{
    std::cout<<NARGS<<" args: \n";
    std::cout<<"  1: num_imgs\n";
    std::cout<<"  2: in_path\n";
    std::cout<<"  3: out_path\n";
    std::cout<<"  4: param_path\n";
    std::cout<<"  5: exec func\n";
    exit(0);
}

int main(int argc, char **argv)
{
    if (argc < NARGS) {usage();}
    const int num_imgs = atoi(argv[1]);
    const char *in_path = argv[2];
    const char *param_path = argv[3];
    const char *out_path = argv[4];
    const int exec = atoi(argv[5]);

    int8_t **in_data = (int8_t**)malloc(num_imgs*sizeof(int8_t*));

    //grab inputs from file
    std::ifstream infile(in_path);

    int count = 0;
    const int input_size = 32*32*3;
    std::cout<<"input size: "<<input_size<<std::endl;
    for (int i=0; i<num_imgs; ++i){
        std::string line;
        std::getline(infile, line);
        std::istringstream ss(line);
        int tmp;
        in_data[i] = (int8_t*)malloc(input_size*sizeof(int8_t));
        for (int j=0; j<input_size; ++j){
          ss >> tmp;
          in_data[i][j] = (int8_t)tmp;
        }
    }
    std::cout<<"done with input data"<<std::endl;

    //layer tensors from file
    Tensor *nodes = tensors_from_file(param_path);
    std::cout<<"done with params\n";

    SpTensor spw[9];
    compress(&nodes[8], &spw[0], 27, 0);
    compress(&nodes[9], &spw[1], 144, 0);
    compress(&nodes[10], &spw[2], 144, 0);
    compress(&nodes[11], &spw[3], 144, 0);
    compress(&nodes[12], &spw[4], 144, 0);
    compress(&nodes[13], &spw[5], 16, 0);
    compress(&nodes[14], &spw[6], 144, 0);
    compress(&nodes[15], &spw[7], 144, 0);
    compress(&nodes[16], &spw[8], 32, 0);


    //run conv on layer inputs
    unsigned long long s, e;
    FILE *fp = fopen(out_path, "w");
    if (fp == NULL) {printf("no output file\n"); exit(EXIT_FAILURE);}
    for (int i=0; i<num_imgs; ++i){
      nodes[0].tensor = in_data[i];
      switch (exec)
      {
      case 0:
        s = read_cycles();
        ScalarDenseConv(1, 1, 1, 1, Relu, &nodes[0], &nodes[8], &nodes[3], &nodes[22]);
        ScalarDenseConv(1, 1, 1, 1, Relu, &nodes[22], &nodes[9], &nodes[4], &nodes[23]);
        ScalarDenseConv(1, 1, 1, 1, None, &nodes[23], &nodes[10], &nodes[17], &nodes[24]);
        AddQuantized(Relu, &nodes[22], &nodes[24], &nodes[25]);
        ScalarDenseConv(1, 1, 2, 2, Relu, &nodes[25], &nodes[11], &nodes[5], &nodes[26]);
        ScalarDenseConv(1, 1, 1, 1, None, &nodes[26], &nodes[12], &nodes[18], &nodes[27]);
        ScalarDenseConv(1, 1, 2, 2, None, &nodes[25], &nodes[13], &nodes[19], &nodes[28]);
        AddQuantized(Relu, &nodes[28], &nodes[27], &nodes[29]);
        ScalarDenseConv(1, 1, 2, 2, Relu, &nodes[29], &nodes[14], &nodes[6], &nodes[30]);
        ScalarDenseConv(1, 1, 2, 2, Relu, &nodes[29], &nodes[14], &nodes[6], &nodes[30]);
        ScalarDenseConv(1, 1, 2, 2, Relu, &nodes[29], &nodes[14], &nodes[6], &nodes[30]);
        ScalarDenseConv(1, 1, 1, 1, None, &nodes[30], &nodes[15], &nodes[20], &nodes[31]);
        ScalarDenseConv(1, 1, 2, 2, None, &nodes[29], &nodes[16], &nodes[21], &nodes[32]);
        AddQuantized(Relu, &nodes[32], &nodes[31], &nodes[33]);
        AveragePool(8, 8, 8, 8, &nodes[33], &nodes[34]);
        nodes[35].tensor = nodes[34].tensor; //flatten
        FullyConnectedDenseScalar(None, &nodes[35], &nodes[7], &nodes[1], &nodes[36]);
        e = read_cycles();
        break;
      case 1:
        s = read_cycles();
        ConvPackedQuantizedDenseCPU(1, 1, 1, 1, Relu, &nodes[0], &nodes[8], &nodes[3], &nodes[22]);
        ConvPackedQuantizedDenseCPU(1, 1, 1, 1, Relu, &nodes[22], &nodes[9], &nodes[4], &nodes[23]);
        ConvPackedQuantizedDenseCPU(1, 1, 1, 1, None, &nodes[23], &nodes[10], &nodes[17], &nodes[24]);
        AddQuantized(Relu, &nodes[22], &nodes[24], &nodes[25]);
        ConvPackedQuantizedDenseCPU(1, 1, 2, 2, Relu, &nodes[25], &nodes[11], &nodes[5], &nodes[26]);
        ConvPackedQuantizedDenseCPU(1, 1, 1, 1, None, &nodes[26], &nodes[12], &nodes[18], &nodes[27]);
        ConvPackedQuantizedDenseCPU(1, 1, 2, 2, None, &nodes[25], &nodes[13], &nodes[19], &nodes[28]);
        AddQuantized(Relu, &nodes[28], &nodes[27], &nodes[29]);
        ConvPackedQuantizedDenseCPU(1, 1, 2, 2, Relu, &nodes[29], &nodes[14], &nodes[6], &nodes[30]);
        ConvPackedQuantizedDenseCPU(1, 1, 2, 2, Relu, &nodes[29], &nodes[14], &nodes[6], &nodes[30]);
        ConvPackedQuantizedDenseCPU(1, 1, 2, 2, Relu, &nodes[29], &nodes[14], &nodes[6], &nodes[30]);
        ConvPackedQuantizedDenseCPU(1, 1, 1, 1, None, &nodes[30], &nodes[15], &nodes[20], &nodes[31]);
        ConvPackedQuantizedDenseCPU(1, 1, 2, 2, None, &nodes[29], &nodes[16], &nodes[21], &nodes[32]);
        AddQuantized(Relu, &nodes[32], &nodes[31], &nodes[33]);
        AveragePool(8, 8, 8, 8, &nodes[33], &nodes[34]);
        nodes[35].tensor = nodes[34].tensor; //flatten
        FullyConnectedDenseScalar(None, &nodes[35], &nodes[7], &nodes[1], &nodes[36]);
        e = read_cycles();
        break;
      case 2:
        s = read_cycles();
        ConvPackedQuantizedSparseCPU(1, 1, 1, 1, Relu, &nodes[0], &spw[0], &nodes[3], &nodes[22]);
        ConvPackedQuantizedSparseCPU(1, 1, 1, 1, Relu, &nodes[22], &spw[1], &nodes[4], &nodes[23]);
        ConvPackedQuantizedSparseCPU(1, 1, 1, 1, None, &nodes[23], &spw[2], &nodes[17], &nodes[24]);
        AddQuantized(Relu, &nodes[22], &nodes[24], &nodes[25]);
        ConvPackedQuantizedSparseCPU(1, 1, 2, 2, Relu, &nodes[25], &spw[3], &nodes[5], &nodes[26]);
        ConvPackedQuantizedSparseCPU(1, 1, 1, 1, None, &nodes[26], &spw[4], &nodes[18], &nodes[27]);
        ConvPackedQuantizedSparseCPU(1, 1, 2, 2, None, &nodes[25], &spw[5], &nodes[19], &nodes[28]);
        AddQuantized(Relu, &nodes[28], &nodes[27], &nodes[29]);
        ConvPackedQuantizedSparseCPU(1, 1, 2, 2, Relu, &nodes[29], &spw[6], &nodes[6], &nodes[30]);
        ConvPackedQuantizedSparseCPU(1, 1, 2, 2, Relu, &nodes[29], &spw[6], &nodes[6], &nodes[30]);
        ConvPackedQuantizedSparseCPU(1, 1, 2, 2, Relu, &nodes[29], &spw[6], &nodes[6], &nodes[30]);
        ConvPackedQuantizedSparseCPU(1, 1, 1, 1, None, &nodes[30], &spw[7], &nodes[20], &nodes[31]);
        ConvPackedQuantizedSparseCPU(1, 1, 2, 2, None, &nodes[29], &spw[8], &nodes[21], &nodes[32]);
        AddQuantized(Relu, &nodes[32], &nodes[31], &nodes[33]);
        AveragePool(8, 8, 8, 8, &nodes[33], &nodes[34]);
        nodes[35].tensor = nodes[34].tensor; //flatten
        FullyConnectedDenseScalar(None, &nodes[35], &nodes[7], &nodes[1], &nodes[36]);
        e = read_cycles();
        break;
      case 3:
        s = read_cycles();
        ConvPackedQuantizedSparseHHT(1, 1, 1, 1, Relu, &nodes[0], &spw[0], &nodes[3], &nodes[22]);
        ConvPackedQuantizedSparseHHT(1, 1, 1, 1, Relu, &nodes[22], &spw[1], &nodes[4], &nodes[23]);
        ConvPackedQuantizedSparseHHT(1, 1, 1, 1, None, &nodes[23], &spw[2], &nodes[17], &nodes[24]);
        AddQuantized(Relu, &nodes[22], &nodes[24], &nodes[25]);
        ConvPackedQuantizedSparseHHT(1, 1, 2, 2, Relu, &nodes[25], &spw[3], &nodes[5], &nodes[26]);
        ConvPackedQuantizedSparseHHT(1, 1, 1, 1, None, &nodes[26], &spw[4], &nodes[18], &nodes[27]);
        ConvPackedQuantizedSparseHHT(1, 1, 2, 2, None, &nodes[25], &spw[5], &nodes[19], &nodes[28]);
        AddQuantized(Relu, &nodes[28], &nodes[27], &nodes[29]);
        ConvPackedQuantizedSparseHHT(1, 1, 2, 2, Relu, &nodes[29], &spw[6], &nodes[6], &nodes[30]);
        ConvPackedQuantizedSparseHHT(1, 1, 2, 2, Relu, &nodes[29], &spw[6], &nodes[6], &nodes[30]);
        ConvPackedQuantizedSparseHHT(1, 1, 2, 2, Relu, &nodes[29], &spw[6], &nodes[6], &nodes[30]);
        ConvPackedQuantizedSparseHHT(1, 1, 1, 1, None, &nodes[30], &spw[7], &nodes[20], &nodes[31]);
        ConvPackedQuantizedSparseHHT(1, 1, 2, 2, None, &nodes[29], &spw[8], &nodes[21], &nodes[32]);
        AddQuantized(Relu, &nodes[32], &nodes[31], &nodes[33]);
        AveragePool(8, 8, 8, 8, &nodes[33], &nodes[34]);
        nodes[35].tensor = nodes[34].tensor; //flatten
        FullyConnectedDenseScalar(None, &nodes[35], &nodes[7], &nodes[1], &nodes[36]);
        e = read_cycles();
        break;
      default:
        std::cout<<"bad exec func\n";
        exit(0);
      }
      fprintf(fp, "%d\n", e-s);

      // int8_t *optr = (int8_t*)nodes[36].tensor;
      // for (int i=0; i<10; ++i){
      //   std::cout<<(int)optr[i]<<" ";
      // }


      // print output for verify
      // for (int oidx=22; oidx<=34; ++oidx){
      //   int8_t *optr = (int8_t*)nodes[oidx].tensor;
      //   int osz = size4D(nodes[oidx].shape);
      //   std::cout<<oidx<<": \n";
      //   for (int i=osz-10; i<osz; ++i){
      //   // for (int i=0; i<10; ++i){
      //     std::cout<<(int)optr[i]<<" ";
      //   }
      //   std::cout<<"\n";
      // }
    }

    return 0;
}