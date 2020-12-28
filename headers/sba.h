#ifndef SBA_H
#define SBA_H

#include <stdint.h>

typedef uint_fast32_t uint;

// sparse bit array. sorted arraylist implementation
typedef struct SBA {
    uint* indices; // contains indices of bits that are ON
    uint size; // number of ON bits in the array
    uint capacity; // mem currently allocated for the list
} SBA;

// initial_cap must be > 0
// returns an empty SBA
SBA* allocSBA(uint initial_cap);

void freeSBA(SBA*);

// reduces the capacity and memory allocated to a, to match its size
void shortenSBA(SBA* a);

// flips bit in array at index to ON
// if the bit is already on, there is no effect (skips duplicate)
void turn_on(SBA* a, uint bit_index);

// flips bit in array at index to OFF
// if the bit is already off, there is no effect
void turn_off(SBA* a, uint bit_index);

// allocates a SBA with sufficient capacity to be used as the result in the AND op.
// the allocated SBA has an uninitalized size, since this is set in the AND op
// this is based on the argument SBAs' CURRENT SIZES, and not their capacities
SBA* allocSBA_and(SBA*, SBA*);

// ANDs a and b, and places the result in r
// r->capacity >= min(a->size, b->size). r can be a or b
void and(SBA* r, SBA* a, SBA* b);

// allocates a SBA with sufficient capacity to be used as the result in the OR op.
// the allocated SBA has an uninitalized size, since this is set in the OR op
// this is based on the argument SBAs' CURRENT SIZES, and not their capacities
SBA* allocSBA_or(SBA*, SBA*);

// ORs a and b, and places the result in r
// r must have a size of 0, and r->capacity >= a->size + b->size. 
// Unlike the AND op, r can't be a or b.
void or(SBA* r, SBA* a, SBA* b);

// increases a by bitshifting n places
void shift(SBA* a, uint n);

// returns 1 if they are equal, and 0 if they are not equal
int equal(SBA* a, SBA* b);

// allocates a SBA with sufficient capacity to be used as the destination in the cp operation.
// this is based on the argument SBA's CURRENT SIZE, and not its capacity
SBA* allocSBA_cp(SBA* src);

// copies the src to the dest
// dest must have sufficient capacity (dest->capacity = src->size)
void cp(SBA* dest, SBA* src);

// randomly flips bits off
// amount is the # of bits that will remain on
// there is a small preference for bits closer to the end of the array to be left on.
SBA* subsample(SBA* a, uint amount);

// randomly flips bits off
// 1 / (2^n) chance of each bit remaining on. n is a power of 2, and must be <= logs(RAND_MAX + 1)
// a's size in memory is left unchanged
void subsample2(SBA* a, uint n);

// randomly flips bits off
// 1 / n chance of each bit remaining on
// a's size in memory is left unchanged
void subsample3(SBA* a, uint n);

// input in [0,1], the value to encode
// n is the number of total bits in the SBA. n >= r->size
// r is an empty sba. r's size is the number of bits to turn on. r's capacity should equal it's size
void encodeLinear(float input, uint n, SBA* r);

// input is the the value to encode. it is encoded linearly, except its encoding wraps back to 0 as it approaches period
// n is the number of total bits in the SBA. n >= r->size
// r is an empty sba. r's size is the number of bits to turn on. r's capacity should equal it's size
void encodePeriodic(float input, float period, uint n, SBA* r);

#endif