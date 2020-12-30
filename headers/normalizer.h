#ifndef NORMALIZER_H
#define NORMALIZER_H

#include "sba.h"

typedef struct Connection  {
    uint_fast8_t active;
    float weight;
} Connection;

// maps inputs to outputs such that:
// - over time, on average, only X% of the output bits will be ON for any inputs
// - similar inputs produce similar outputs
// - disimilar inputs produce disimilar outputs
typedef struct Normalizer {
    uint inputSize;
    uint outputSize;
    float sparsity;
    // TODO compare different thresholds to enforce sparcity
    //   - thresholds per output
    //   - single thresholds for entire output <-- current implementation
    float threshold;
    float thresholdDelta;
    // connections to outputs from same inputs are contiguous
    // connection_index = outputBit + inputBit * n->outputSize;
    Connection* connections;
    float weightDelta;
} Normalizer;

Normalizer* allocNormalizer(uint inputSize, uint outputSize, float sparsity, float threshold, float thresholdDelta, float weightDelta);

void freeNormalizer(Normalizer* n);

// Sets n's connections. Each connection has an activeLikelyhood rate of being set to active
void randomizeNormalizer(Normalizer* n, float activeLikelyhood);

// allocates a SBA with sufficient capacity to be used as the output of a normalizer
// the allocated SBA has an uninitalized size, since this is set in the output function
SBA* allocSBA_norm_output(Normalizer* n);

// fills the output based on the input, and self-adjusts the threshold and connection strengths
// the input and output can be the same SBA, as long as they have sufficient capacity
void getNormalizerOutput(SBA* output, Normalizer* n, SBA* input);

#endif