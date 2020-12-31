#ifndef SBA_MAPPER_H
#define SBA_MAPPER_H

#include "sba.h"

typedef struct SBAMapperOutput {
    // each output is connected to a random subsample of the inputs,
    // and each connection has a strength
    uint numConnections;
    uint* inputBits;
    uint8_t* strengths;
} SBAMapperOutput;

// maps inputs to outputs such that:
// - only X% of the output bits will be ON for any inputs
// - similar inputs produce similar outputs
// - disimilar inputs produce disimilar outputs
typedef struct SBAMapper {
    uint numOutputs;
    SBAMapperOutput* outputs;
    uint numActiveOutputs;
    uint8_t connectionStrengthThreshold;
    uint8_t connectionStrengthDelta;
} SBAMapper;

// returns a randomly initalized SBAMapper
SBAMapper* allocSBAMapper(uint numInputs,
    uint numOutputs,
    float connectionLikelihood,
    uint numActiveOutputs,
    uint8_t connectionStrengthThreshold,
    uint8_t connectionStrengthDelta);

void freeSBAMapper(SBAMapper*);

void printSBAMapper(SBAMapper*);

// allocates a SBA with sufficient capacity to be used as the output of the doMapper function
// the allocated SBA has an uninitalized size, since this is set in the function
SBA* allocSBA_mapper_output(SBAMapper* m);

void doMapper(SBA* output, SBAMapper* m, SBA* input);

#endif