#ifndef SBA_H
#define SBA_H

#include <stdint.h>

typedef uint64_t uint;

// sparse bit array. sorted arraylist implementation
typedef struct SBA {
    uint* indices; // contains indices of bits that are ON
    uint size; // number of ON bits in the array
    uint capacity; // mem currently allocated for the list
} SBA;

// leaves size uninitialized
SBA* _allocSBA_nosetsize(uint initialCap);

// initial_cap must be > 0
// returns an empty SBA
SBA* allocSBA(uint initialCap);

void freeSBA(SBA*);

void printSBA(SBA*);

// reduces the capacity and memory allocated to a, to match its size
void shortenSBA(SBA* a);

// flips bit in array at index to ON
// if the bit is already on, there is no effect (skips duplicate)
void turnOn(SBA* a, uint bitIndex);

// flips bit in array at index to OFF
// if the bit is already off, there is no effect
void turnOff(SBA* a, uint bitIndex);

// returns bool state of bit at index
uint8_t getBit(SBA* a, uint bitIndex);

// turns bits in a to off that are also contained in rm
void turnOffAll(SBA* a, SBA* rm);

// allocates a SBA with sufficient capacity to be used as the result in the AND op.
// the allocated SBA has an uninitalized size, since this is set in the AND op
// this is based on the argument SBAs' CURRENT SIZES, and not their capacities
SBA* allocSBA_andBits(SBA*, SBA*);

// ANDs a and b, and places the result in r
// r->capacity >= min(a->size, b->size). r can be a or b
void andBits(SBA* r, SBA* a, SBA* b);

// returns the number of bits on in a AND b
uint andSize(SBA* a, SBA* b);

// allocates a SBA with sufficient capacity to be used as the result in the OR or XOR op.
// the allocated SBA has an uninitalized size, since this is set in the OR op
// this is based on the argument SBAs' CURRENT SIZES, and not their capacities
SBA* allocSBA_or(SBA*, SBA*);

// ORs a and b, and places the result in r
// r->capacity >= a->size + b->size. 
// Unlike the AND op, r can't be a or b.
// if exclusive is nonzero, XOR is used instead
void orBits(SBA* r, SBA* a, SBA* b, uint8_t exclusive);

// returns the number of bits on in a OR b. if exclusive is nonzero, XOR is used instead
uint orSize(SBA* a, SBA* b, uint8_t exclusive);

// increases a by bitshifting n places
void rshift(SBA* a, uint n);

// decreases a by bitshifting n places
void lshift(SBA* a, uint n);

// returns 1 if they are equal, and 0 if they are not equal
uint8_t equal(SBA* a, SBA* b);

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

// returns a subsample of a, with a rate of retaining each bit
SBA* subsample4(SBA* a, float rate);

// input in [0,1], the value to encode
// n is the number of total bits in the SBA. n >= r->size
// r is an empty sba. r's size is the number of bits to turn on. r's capacity should equal it's size
void encodeLinear(float input, uint n, SBA* r);

// input is the the value to encode. it is encoded linearly, except its encoding wraps back to 0 as it approaches period
// n is the number of total bits in the SBA. n >= r->size
// r is an empty sba. r's size is the number of bits to turn on. r's capacity should equal it's size
void encodePeriodic(float input, float period, uint n, SBA* r);

#endif