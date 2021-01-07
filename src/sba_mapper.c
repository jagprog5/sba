#include "sba_mapper.h"

#include <time.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

SBAMapper* allocSBAMapper(uint32_t numInputs,
        uint32_t numOutputs,
        float connectionLikelihood,
        uint32_t numActiveOutputs,
        uint8_t connectionStrengthThreshold) {
    SBAMapper* m = malloc(sizeof(*m) + sizeof(m->connections[0]) * numOutputs);
    m->numActiveOutputs = numActiveOutputs;
    m->numOutputs = numOutputs;
    m->connectionStrengthThreshold = connectionStrengthThreshold;

    SBAConnection *connections = malloc(sizeof(*connections) * ((numInputs + 1) * numOutputs));
    uint_fast32_t connectionLikelihoodVal = connectionLikelihood * RAND_MAX;
    uint_fast32_t connectionIndex = 0;
    for (uint_fast32_t i = 0; i < numOutputs; ++i) {
        for (uint_fast32_t j = 0; j < numInputs; ++j) {
            if (rand() < connectionLikelihoodVal) {
                connections[connectionIndex++] = (SBAConnection){j, rand() % UINT8_MAX};
            }
        }
        connections[connectionIndex++] = (SBAConnection){M_SENTINEL_VAL, UINT8_MAX};
    }
    connections = realloc(connections, connectionIndex * sizeof(*connections));

    connectionIndex = 0;
    uint_fast8_t prev_was_sentinel = 1;
    for (uint_fast32_t i = 0; i < numOutputs; ++i) {
        while (!prev_was_sentinel) {
            prev_was_sentinel = connections[connectionIndex++].inputBit == M_SENTINEL_VAL;
        }
        prev_was_sentinel = 0;
        m->connections[i] = connections + connectionIndex;
    }
    return m;
}

void freeSBAMapper(SBAMapper* m) {
    free(m->connections[0]);
    free(m);
}

void printSBAMapper(SBAMapper* m) {
    puts("=====SBAMapper=====");
    printf("numOutputs: %d\n", m->numOutputs);
    printf("numActiveOutputs: %d\n", m->numActiveOutputs);
    printf("connectionStrengthThreshold: %d\n", m->connectionStrengthThreshold);
    for (int_fast32_t i = 0; i < m->numOutputs; ++i) {
        SBAConnection *cptr = m->connections[i];
        printf("Output %ld:\n", i);
        SBAConnection c;
        while ((c = *(cptr++)).inputBit != M_SENTINEL_VAL) {
            printf("\ti:%d, s:%d\n", c.inputBit, c.strength);
        }
    }
}

SBA* allocSBA_doMapper(SBAMapper* m) {
    return _allocSBA_nosetsize(m->numActiveOutputs);
}

typedef struct {
    uint_fast32_t index;
    uint_fast32_t value;
} TwoTuple;

int _cmp_by_index(const void * a, const void * b) {
   return ((TwoTuple*)a)->index - ((TwoTuple*)b)->index;
}

void doMapper(SBA* output, SBAMapper* m, SBA* input, uint8_t connectionStrengthDelta) {
    uint_fast32_t numOutputs = m->numOutputs;
    uint_fast8_t threshold = m->connectionStrengthThreshold;
    uint_fast32_t input_arr_size = input->size;
    uint_fast32_t outputScores[numOutputs];
    memset(outputScores, 0, sizeof(outputScores));
    for (uint_fast32_t outputIndex = 0; outputIndex < numOutputs; ++outputIndex) {
        SBAConnection *cptr = m->connections[outputIndex];
        SBAConnection c;
        uint_fast32_t input_arr_index = 0;
        uint_fast32_t input_arr_value;
        uint_fast8_t input_arr_prev_value_valid = 0;
        while ((c = *(cptr++)).inputBit != M_SENTINEL_VAL) {
            uint_fast32_t inputIndex = c.inputBit;
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
                uint_fast8_t strength = c.strength;
                if (strength > threshold) {
                    // if the input bit is ON,
                    // and there is a connection to this output,
                    // and the strength is sufficient
                    // then increase the score for this output bit
                    outputScores[outputIndex] += 1;
                }
            } else if (input_arr_value < inputIndex) {
                goto next_input_arr_bit;
            } else {
                input_arr_prev_value_valid = 1;
            }
        }
    }
    // =============================================================================
    // TODO decrease output scores for outputs that are active too often, and increase for inactive outputs
    // Moving average of some sort...

    // =========================================================================
    // at this point we have the output scores
    // get the top n scores' indices, in order, from an unsorted list, and place in output
    uint_fast32_t numActiveOutputs = m->numActiveOutputs;
    uint_fast32_t sizeActiveOutputs = 0;
    TwoTuple activeOutputs[numActiveOutputs];
    for (uint_fast32_t i = 0; i < numOutputs; ++i) {
        uint_fast32_t score = outputScores[i];
        int64_t left = 0;
        int64_t right = sizeActiveOutputs - 1;
        int64_t middle = 0;
        uint_fast32_t mid_val = 0;
        while (left <= right) {
            middle = (right + left) / 2;
            mid_val = activeOutputs[middle].value;
            if (mid_val > score) {
                left = middle + 1;
            } else if (mid_val < score) {
                right = middle - 1;
            } else {
                break;
            }
        }
        if (score < mid_val) {
            middle += 1;
        }
        uint_fast32_t tuplesToMove;
        if (sizeActiveOutputs >= numActiveOutputs) {
            // array is full
            if (middle == numActiveOutputs) {
                continue; // skip appending to end of array. Also, can't have memmove w/ negative n
            }
            tuplesToMove = numActiveOutputs - 1 - middle;
        } else {
            // array can expand
            tuplesToMove = sizeActiveOutputs++ - middle;
        }
        memmove(activeOutputs + middle + 1, activeOutputs + middle, sizeof(TwoTuple) * tuplesToMove);
        activeOutputs[middle] = (TwoTuple){i, score};
    }
    qsort(activeOutputs, sizeActiveOutputs, sizeof(TwoTuple), _cmp_by_index);
    for (uint_fast32_t i = 0; i < sizeActiveOutputs; ++i) {
        output->indices[i] = activeOutputs[i].index;
    }
    output->size = sizeActiveOutputs;
    // =============================================================================
    // Do training
    if (connectionStrengthDelta == 0) return;
    for (uint_fast32_t i = 0; i < sizeActiveOutputs; ++i) {
        SBAConnection *cptr = m->connections[activeOutputs[i].index];
        SBAConnection c;
        uint_fast32_t input_arr_index = 0;
        uint_fast32_t input_arr_value;
        uint_fast8_t input_arr_prev_value_valid = 0;
        while ((c = *cptr).inputBit != M_SENTINEL_VAL) {
            uint_fast32_t inputIndex = c.inputBit;
            uint8_t strength;
            // check if the input is on for this connection
            // if it is, increment the strength, else, decrement
            next_input_arr_bit2:
            if (!input_arr_prev_value_valid) {
                if (input_arr_index >= input_arr_size) {
                    input_arr_value = 0;
                    // the input bit has not yet been found and the end of the inputs has been reached
                    goto input_known;
                }
            } else {
                input_arr_prev_value_valid = 0;
            }
            input_arr_value = input->indices[input_arr_index++];
            if (input_arr_value == inputIndex) {
                input_arr_value = 1;
            } else if (input_arr_value < inputIndex) {
                goto next_input_arr_bit2;
            } else {
                input_arr_prev_value_valid = 1;
                input_arr_value = 0;
            }

            input_known:
            strength = c.strength;
            if (input_arr_value) {
                if (strength <= UINT8_MAX - connectionStrengthDelta) {
                    cptr->strength = strength + connectionStrengthDelta;
                }
            } else {
                if (strength >= connectionStrengthDelta) {
                    cptr->strength = strength - connectionStrengthDelta;
                }
            }
            cptr += 1;
        }
    }
}

int main() {
    srand(time(NULL));
    SBAMapper* m = allocSBAMapper(3, 3, 1.0f, 2, 0);
    printSBAMapper(m);
    SBA* input = allocSBA(0);
    turnOn(&input, 0);
    turnOn(&input, 1);
    turnOn(&input, 2);
    SBA* output = allocSBA_doMapper(m);
    doMapper(output, m, input, 1);
    printSBA(output);
    printSBAMapper(m);
    return 0;
}