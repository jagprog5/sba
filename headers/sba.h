#ifndef SBA_H
#define SBA_H

#include <stdint.h>

/*
 * TODO create in_place functions to ease strain on malloc
 * - Write result SBA into first argument?
 * - Require pre-allocated ptr to SBA of proper size to write to?
 */

typedef uint_fast32_t uint;

// sparse bit array. sorted arraylist implementation
typedef struct SBA {
    uint* indices; // contains indices of bits that are ON
    uint size; // number of ON bits in the array
    uint capacity; // mem currently allocated for the list
} SBA;

// initial_cap should be a power of 2
SBA* allocSBA(uint inital_cap);

void freeSBA(SBA*);

// flips bit in array at index to ON
// if the bit is already on, there is no effect (skips duplicate)
void insert(SBA* a, uint bit_index);

// returns AND of two SBAs, as ptr to heap
SBA* and(SBA* a, SBA* b);

// returns OR of two SBAs, as ptr to heap
SBA* or(SBA* a, SBA* b);

// randomly flips bits off in returned SBA
// amount is the # of bits that will remain on
// there is a small preference for bits closer to the end of the array to be left on.
SBA* subsample(SBA* a, uint amount);

// input in [0,1]
// width is the number of total bits in the SBA. width >= r->size
// r is an empty sba. r's size is the number of bits to turn on. r's capacity should equal it's size
void encodeLinear(float input, uint width, SBA* r);

#endif