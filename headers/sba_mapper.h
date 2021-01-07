#ifndef SBA_MAPPER_H
#define SBA_MAPPER_H

#include "sba.h"

#define M_SENTINEL_VAL UINT32_MAX

typedef struct SBAConnection {
    uint32_t inputBit;
    uint8_t strength;
} SBAConnection;

// maps inputs to outputs such that:
// - only n of the output bits will be ON for any inputs
// - in the long term, similar inputs produce similar outputs
// - in the long term, disimilar inputs produce disimilar outputs
typedef struct SBAMapper {
    uint32_t numActiveOutputs;
    uint32_t numOutputs;
    uint8_t connectionStrengthThreshold;
    SBAConnection *connections[];
} SBAMapper;

// returns a randomly initalized SBAMapper
// connectionLikelihood in range [0,1], the likelihood of an output connection to each input
// numActiveOutputs is the number of outputs that will be 1 in the doMapper function
// connectionStrengthThreshold is the threshold that a connection's strength must exceed to be effective in influencing the output
// connectionStrengthDelta should be a small value like 1 or 2
SBAMapper* allocSBAMapper(uint32_t numInputs,
    uint32_t numOutputs,
    float connectionLikelihood,
    uint32_t numActiveOutputs,
    uint8_t connectionStrengthThreshold);

void freeSBAMapper(SBAMapper*);

void printSBAMapper(SBAMapper*);

// allocates a SBA with sufficient capacity to be used as the output of the doMapper function
// the allocated SBA has an uninitalized size, since this is set in the function
SBA* allocSBA_doMapper(SBAMapper* m);

void doMapper(SBA* output, SBAMapper* m, SBA* input, uint8_t connectionStrengthDelta);

#endif