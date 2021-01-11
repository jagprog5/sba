#ifndef SBA_H
#define SBA_H

#include <stdint.h>

// sparse bit array. sorted arraylist implementation
typedef struct SBA {
    uint32_t size; // number of ON bits in the array
    uint32_t capacity; // mem currently allocated for the list
    uint32_t* indices; // contains indices of bits that are ON.
} SBA;

// leaves size uninitialized
SBA* _allocSBA_nosetsize(uint32_t initialCap);

// returns an empty SBA
SBA* allocSBA(uint32_t initialCap);

void freeSBA(SBA*);

void printSBA(SBA*);

// reduces the capacity and memory allocated to a, to match its size
void shortenSBA(SBA* a);

// flips bit in array at index to ON
// if the bit is already on, there is no effect (skips duplicate)
// reallocs if sufficiently large
void turnOn(SBA* a, uint32_t bitIndex);

// flips bit in array at index to OFF
// if the bit is already off, there is no effect
// reallocs if sufficiently small
void turnOff(SBA* a, uint32_t bitIndex);

// returns bool state of bit at index
uint8_t getBit(SBA* a, uint32_t bitIndex);

// turns bits in a to off that are also contained in rm
void turnOffAll(SBA* a, SBA* rm);

// allocates an SBA with sufficient capacity to be used as the result in the AND op.
// the allocated SBA has an uninitalized size, since this is set in the AND op
// this is based on the argument SBAs' CURRENT SIZES, and not their capacities
SBA* allocSBA_andBits(SBA*, SBA*);

// If size_only is false:
//      ANDs a and b, and places the result in r, an SBA pointer.
//      r->capacity >= min(a->size, b->size). r can be a or b.
// If size_only is true:
//      ANDs a and b, and writes the number of bits in common to the uint32_t* r.
void andBits(void* r, SBA* a, SBA* b, uint8_t size_only);

// allocates an SBA with sufficient capacity to be used as the result in the OR or XOR op.
// the allocated SBA has an uninitalized size, since this is set in the OR op.
// this is based on the argument SBAs' CURRENT SIZES, and not their capacities.
SBA* allocSBA_or(SBA*, SBA*);

// If size_only is false:
//      ORs a and b, and places the result in r, an SBA pointer.
//      r->capacity >= a->size + b->size. 
//      Unlike the AND op, r can't be a or b.
// If size_only is true:
//      ORs a and b, and writes the number of bits in common to the uint32_t* r.
// if exclusive is true, XOR is used instead of OR.
void orBits(void* r, SBA* a, SBA* b, uint8_t exclusive, uint8_t size_only);

// increases a by bitshifting n places
void rshift(SBA* a, uint32_t n);

// decreases a by bitshifting n places
void lshift(SBA* a, uint32_t n);

// returns 1 if they are equal, and 0 if they are not equal
uint8_t equal(SBA* a, SBA* b);

// allocates an SBA with sufficient capacity to be used as the destination in the cp operation.
// this is based on the argument SBA's CURRENT SIZE, and not its capacity
SBA* allocSBA_cp(SBA* src);

// copies the src to the dest
// dest must have sufficient capacity (dest->capacity = src->size)
void cp(SBA* dest, SBA* src);

// randomly flips bits off
// amount in range [0, 1], where 0 clears the list
void subsample(SBA* a, float amount);

// input in [0,1], the value to encode
// n is the number of total bits in the SBA. n >= r->size
// r is an empty sba. r's size is the number of bits to turn on. r's capacity should equal it's size
void encodeLinear(float input, uint32_t n, SBA* r);

// input is the the value to encode. it is encoded linearly, except its encoding wraps back to 0 as it approaches period
// n is the number of total bits in the SBA. n >= r->size
// r is an empty sba. r's size is the number of bits to turn on. r's capacity should equal it's size
void encodePeriodic(float input, float period, uint32_t n, SBA* r);

#endif