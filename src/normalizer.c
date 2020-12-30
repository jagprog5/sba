#include "normalizer.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

Normalizer* allocNormalizer(uint inputSize, uint outputSize, float sparsity, float threshold, float thresholdDelta, float weightDelta) {
    Normalizer* n = malloc(sizeof(*n));
    n->inputSize = inputSize;
    n->outputSize = outputSize;
    n->sparsity = sparsity;
    n->threshold = threshold;
    n->thresholdDelta = thresholdDelta;
    n->connections = malloc(sizeof(*n->connections) * inputSize * outputSize);
    n->weightDelta = weightDelta;
    return n;
}

void freeNormalizer(Normalizer* n) {
    free(n->connections);
    free(n);
}

void printNormalizer(Normalizer* n) {
    uint outputOffset = 0;
    uint num_inputs = n->inputSize;
    uint num_outputs = n->outputSize;
    for (uint inputBit = 0; inputBit < num_inputs; ++inputBit) {
        for (uint outputBit = 0; outputBit < num_outputs; ++outputBit) {
            Connection* c = n->connections + outputBit + outputOffset;
            printf("i:%03ld o:%03ld c:%d*%.3f\n", inputBit, outputBit, c->active, c->weight);
        }
        outputOffset += num_outputs;
    }
}

void randomizeNormalizer(Normalizer* n, float activeLikelyhood) {
    uint numCon = n->inputSize * n->outputSize;
    for (uint i = 0; i < numCon; ++i) {
        Connection* c = n->connections + i;
        if ((float)rand() / RAND_MAX < activeLikelyhood) {
            c->active = 1;
            c->weight = (float)rand() / RAND_MAX;
        } else {
            c->active = 0;
        }
    }
}

SBA* allocSBA_norm_output(Normalizer* n) {
    return _allocSBA_nosetsize(n->outputSize);
}

// there is a connection between every output bit and input bit.
//     If the connection is active, and the input is active, the weight is added to the output.
//     If the output meets the threshold, then the output is ON.
void getNormalizerOutput(SBA* output, Normalizer* n, SBA* input) {
    uint outputOffset = 0;
    uint num_inputs = n->inputSize;
    uint num_outputs = n->outputSize;
    float output_arr_accums[num_outputs];
    memset(output_arr_accums, 0, sizeof(output_arr_accums));
    uint input_arr_size = input->size;
    uint input_arr_index = 0;
    uint input_arr_value;
    uint_fast8_t input_arr_prev_value_valid = 0;
    for (uint inputBit = 0; inputBit < num_inputs; ++inputBit) {
        next_input_arr_bit:
        if (!input_arr_prev_value_valid) {
            if (input_arr_index >= input_arr_size) {
                break;
            }
            input_arr_value = input->indices[input_arr_index++];
        } else {
            input_arr_prev_value_valid = 0;
        }
        if (input_arr_value == inputBit) {
            for (uint outputBit = 0; outputBit < num_outputs; ++outputBit) {
                Connection* c = n->connections + outputBit + outputOffset;
                if (c->active) {
                    output_arr_accums[outputBit] += c->weight;
                }
            }
        } else if (input_arr_value < inputBit) {
            goto next_input_arr_bit;
        } else if (input_arr_value > inputBit) {
            input_arr_prev_value_valid = 1;
        }
        outputOffset += num_outputs;
    }

    uint output_arr_size = 0;
    for (uint i = 0; i < num_outputs; ++i) {
        if (output_arr_accums[i] > n->threshold) {
            output->indices[output_arr_size++] = i;
        }
    }
    output->size = output_arr_size;
}


int main() {
    srand(time(NULL));
    Normalizer* n = allocNormalizer(3, 2, 0.01f, 0.5f, 0.01f, 0.01f);
    randomizeNormalizer(n, 1);
    printNormalizer(n);
    SBA* input = allocSBA(3);
    turnOn(input, 1);
    // turnOn(input, 2);
    // SBA* output = allocSBA_norm_output(n);
    getNormalizerOutput(input, n, input);
    printSBA(input);
    return 0;
}