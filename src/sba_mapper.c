#include "sba_mapper.h"

#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

SBAMapper* allocSBAMapper(uint numInputs,
        uint numOutputs,
        float connectionLikelihood,
        uint numActiveOutputs,
        uint_fast8_t connectionStrengthThreshold,
        uint_fast8_t connectionStrengthDelta) {
    SBAMapper* m = malloc(sizeof(*m));
    m->numOutputs = numOutputs;
    m->outputs = malloc(sizeof(*m->outputs) * numOutputs);
    for (int i = 0; i < numOutputs; ++i) {
        uint inputsConnected = 0;
        uint inputBits[numInputs];
        uint_fast8_t strengths[numInputs];
        for (int j = 0; j < numInputs; ++j) {
            if ((float)rand() / RAND_MAX < connectionLikelihood) {
                inputBits[inputsConnected] = j;
                strengths[inputsConnected++] = rand() % UINT_FAST8_MAX;
            }
        }
        SBAMapperOutput* mo = m->outputs + i;
        mo->numConnections = inputsConnected;
        mo->inputBits = malloc(sizeof(*mo->inputBits) * inputsConnected);
        mo->strengths = malloc(sizeof(*mo->strengths) * inputsConnected);
        memcpy(mo->inputBits, inputBits, sizeof(*mo->inputBits) * inputsConnected);
        memcpy(mo->strengths, strengths, sizeof(*mo->strengths) * inputsConnected);
    }
    m->numActiveOutputs = numActiveOutputs;
    m->connectionStrengthThreshold = connectionStrengthThreshold;
    m->connectionStrengthDelta = connectionStrengthDelta;
    return m;
}

void freeSBAMapper(SBAMapper* m) {
    for (int i = 0; i < m->numOutputs; ++i) {
        SBAMapperOutput* mo = m->outputs + i;
        free(mo->inputBits);
        free(mo->strengths);
    }
    free(m->outputs);
    free(m);
}

void printSBAMapper(SBAMapper* m) {
    puts("=====SBAMapper=====");
    printf("numOutputs: %ld\n", m->numOutputs);
    printf("numActiveOutputs: %ld\n", m->numActiveOutputs);
    printf("connectionStrengthThreshold: %d\n", m->connectionStrengthThreshold);
    printf("connectionStrengthDelta: %d\n", m->connectionStrengthDelta);
    for (uint i = 0; i < m->numOutputs; ++i) {
        printf("Output %ld:\n", i);
        SBAMapperOutput* mo = m->outputs + i;
        for (int j = 0; j < mo->numConnections; ++j) {
            printf("\ti:%ld, s:%d\n", mo->inputBits[j], mo->strengths[j]);
        }
    }
}

SBA* allocSBA_mapper_output(SBAMapper* m) {
    return _allocSBA_nosetsize(m->numActiveOutputs);
}

typedef struct {
    uint index;
    uint value;
} TwoTuple;

void doMapper(SBA* output, SBAMapper* m, SBA* input) {
    uint numOutputs = m->numOutputs;
    uint8_t threshold = m->connectionStrengthThreshold;
    uint input_arr_size = input->size;
    uint outputScores[numOutputs];
    memset(outputScores, 0, sizeof(outputScores));
    for (int outputIndex = 0; outputIndex < numOutputs; ++outputIndex) {
        SBAMapperOutput* mo = m->outputs + outputIndex;
        uint numConnections = mo->numConnections;
        uint input_arr_index = 0;
        uint input_arr_value;
        uint_fast8_t input_arr_prev_value_valid = 0;
        for (int conectionIndex = 0; conectionIndex < numConnections; ++conectionIndex) {
            uint inputIndex = mo->inputBits[conectionIndex];
            next_input_arr_bit:
            if (!input_arr_prev_value_valid) {
                if (input_arr_index >= input_arr_size) {
                    break;
                }
                input_arr_value = input->indices[input_arr_index++];
            } else {
                input_arr_prev_value_valid = 0;
            }
            if (input_arr_value == inputIndex) {
                uint strength = mo->strengths[conectionIndex];
                if (strength > threshold) {
                    // if the input bit is ON,
                    // and there is a connection to this output,
                    // and the strength is sufficient
                    // then increase the score for this output bit
                    outputScores[outputIndex] += 1;
                }
            } else if (input_arr_value < inputIndex) {
                goto next_input_arr_bit;
            } else if (input_arr_value > inputIndex) {
                input_arr_prev_value_valid = 1;
            }
        }
    }
    // =========================================================================
    // get the top n indices, in order, from an unsorted list, and place in output
    uint numActiveOutputs = m->numActiveOutputs;
    uint sizeActiveOutputs = 0;
    TwoTuple activeOutputs[numActiveOutputs];
    for (uint i = 0; i < numOutputs; ++i) {
        uint score = outputScores[i];
        printf("%ld %ld\n", i, score);
        uint_fast8_t pushing = 0;
        TwoTuple pushVal;
        for (uint j = 0; j < sizeActiveOutputs; ++j) {
            if (!pushing) {
                if (score <= activeOutputs[j].value) {
                    continue;
                }
                pushing = 1;
                pushVal = (TwoTuple){i, score};
            }
            TwoTuple tmp =  activeOutputs[j];
            activeOutputs[j] = pushVal;
            pushVal = tmp;
        }
        if (sizeActiveOutputs < numActiveOutputs) {
            activeOutputs[sizeActiveOutputs++] = pushing ? pushVal : (TwoTuple){i, score};
        }
    }
    for (int i = 0; i < sizeActiveOutputs; ++i) {
        output->indices[i] = activeOutputs[i].index;
    }
    output->size = sizeActiveOutputs;
}

int main() {
    srand(time(NULL));
    SBAMapper* m = allocSBAMapper(3, 3, 0.5f, 2, 100, 1);
    printSBAMapper(m);
    SBA* input = allocSBA(3);
    turnOn(input, 0);
    turnOn(input, 1);
    turnOn(input, 2);
    SBA* output = allocSBA_mapper_output(m);
    doMapper(output, m, input);
    printSBA(output);
    return 0;
}