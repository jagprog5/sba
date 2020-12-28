#include "sba.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

SBA* _allocSBA_nosetsize(uint initial_cap) {
    SBA* a = malloc(sizeof(*a));
    a->indices = malloc(sizeof(uint) * initial_cap);
    a->capacity = initial_cap;
    return a;
}

SBA* allocSBA(uint initial_cap) {
    SBA* a = _allocSBA_nosetsize(initial_cap);
    a->size = 0;
    return a;
}

void shortenSBA(SBA* a) {
    a->capacity = a->size > 0 ? a->size : 1;
    a->indices = realloc(a->indices, sizeof(uint) * a->capacity);
}

void freeSBA(SBA* a) {
    free(a->indices);
    free(a);
}

void print(SBA* a) { // debug / testing purposes
    if (!a->size) {
        puts("SBA is empty!");
        return;
    }
    for (int i = 0; i < a->size - 1; ++i) {
        printf("  ");
    }
    puts("V");
    for (int i = 0; i < a->capacity; ++i) {
        printf("%d ", (int)a->indices[i]);
    }
    putchar('\n');
}

void turn_on(SBA* a, uint bit_index) {
    if (a->size >= a->capacity) {
        a->capacity <<= 1;
        if (a->capacity == 0) {
            a->capacity = 1;
        }
        a->indices = realloc(a->indices, sizeof(uint) * a->capacity);
    }
    int_fast32_t left = 0;
    int_fast32_t right = a->size - 1;
    int_fast32_t middle;
    uint mid_val = UINT_FAST32_MAX;
    while (left <= right) {
        middle = (right + left) / 2;
        mid_val = a->indices[middle];
        if (mid_val < bit_index) {
            left = middle + 1;
        } else if (mid_val > bit_index) {
            right = middle - 1;
        } else {
            return; // skip duplicate
        }
    }
    if (bit_index > mid_val) {
        middle += 1;
    }
    memmove(a->indices + middle + 1, a->indices + middle, sizeof(uint) * (a->size - middle));
    a->size += 1;
    a->indices[middle] = bit_index;
}

void turn_off(SBA* a, uint bit_index) {
    int_fast32_t right = a->size - 1;
    int_fast32_t left = 0;
    int_fast32_t middle;
    while (left <= right) {
        middle = (right + left) / 2;
        uint mid_val = a->indices[middle];
        if (mid_val == bit_index) {
            a->size -= 1;
            memmove(a->indices + middle, a->indices + middle + 1, sizeof(uint) * (a->size - middle));
            return;
        } else if (mid_val < bit_index) {
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
}

SBA* allocSBA_and(SBA* a, SBA* b) {
    return _allocSBA_nosetsize(a->size < b->size ? a->size : b->size); // and sets the size
}

void and(SBA* r, SBA* a, SBA* b) {
    uint a_offset = 0;
    uint a_val;
    uint a_size = a->size; // store in case r = a
    uint b_offset = 0;
    uint b_val;
    uint b_size = b->size; // store in case r = b
    uint r_size = 0;
    get_both:
    if (a_offset >= a_size) {
        goto end;
    }
    a_val = a->indices[a_offset++];
    if (b_offset >= b_size) {
        goto end;
    }
    b_val = b->indices[b_offset++];

    loop:
    if (a_val < b_val) {
        // get a
        if (a_offset >= a_size) {
            goto end;
        }
        a_val = a->indices[a_offset++];
        goto loop;
    } else if (a_val == b_val) {
        r->indices[r_size++] = a_val;
        goto get_both;
    } else {
        // get b
        if (b_offset >= b_size) {
            goto end;
        }
        b_val = b->indices[b_offset++];
        goto loop;
    }
    end:
    r->size = r_size;
}

SBA* allocSBA_or(SBA* a, SBA* b) {
    return _allocSBA_nosetsize(a->size + b->size);
}

void or(SBA* r, SBA* a, SBA* b) {
    uint a_offset = 0;
    uint a_val;
    uint b_offset = 0;
    uint b_val;
    uint r_size = 0;

    get_both:
    if (a_offset >= a->size) {
        if (!b) {
            goto end;
        }
        a = NULL;
        goto get_b;
    }
    a_val = a->indices[a_offset++];
    if (b_offset >= b->size) {
        if (!a) {
            goto end;
        }
        b = NULL;
        goto loop;
    }
    b_val = b->indices[b_offset++];

    loop:
    if ((a && b && a_val < b_val) || (a && !b)) {
        r->indices[r_size++] = a_val;
        goto get_a;
    } else if ((a && b && a_val > b_val) || (!a && b)) {
        r->indices[r_size++] = b_val;
        goto get_b;
    } else if (a && b && a_val == b_val) {
        r->indices[r_size++] = a_val;
        goto get_both;
    }

    get_a:
    if (a_offset >= a->size) {
        if (!b) {
            goto end;
        }
        a = NULL;
        goto loop;
    }
    a_val = a->indices[a_offset++];
    goto loop;

    get_b:
    if (b_offset >= b->size) {
        if (!a) {
            goto end;
        }
        b = NULL;
        goto loop;
    }
    b_val = b->indices[b_offset++];
    goto loop;

    end:
    r->size = r_size;
}

void shift(SBA* a, uint n) {
    for (int i = 0; i < a->size; ++i) {
        a->indices[i] += n;
    }
}

int equal(SBA* a, SBA* b) {
    if (a->size != b->size) {
        return 0;
    }
    for (int i = 0; i < a->size; ++i) {
        if (a->indices[i] != b->indices[i]) {
            return 0;
        }
    }
    return 1;
}

SBA* allocSBA_cp(SBA* a) {
    return allocSBA(a->size);
}

void cp(SBA* dest, SBA* src) {
    dest->size = src->size;
    memcpy(dest->indices, src->indices, sizeof(uint) * dest->size);
}

SBA* subsample(SBA* a, uint amount) {
    SBA* s = allocSBA(amount);
    uint chunk_index = 0;
    uint chunk_size = a->size / amount;
    uint chunk_size_remainder = a->size % amount;
    uint this_chunk_size;

    loop:
    this_chunk_size = chunk_size + (chunk_size_remainder-- > 0);
    // TODO better rand implementation. Reuse different bits before next rand()?
    // TODO better chunk placement. Smaller chunks are currently all at the end.
    s->indices[s->size++] = a->indices[chunk_index + rand() % this_chunk_size];
    chunk_index += this_chunk_size;
    if (chunk_index >= a->size) {
        return s;
    }
    goto loop;
}

void subsample2(SBA* a, uint n) {
    static uint num_bits = sizeof(int) * 8 - __builtin_clz(RAND_MAX);
    uint bits_taken = num_bits;
    uint and_mask = ((uint)1 << n) - 1;
    uint to_offset = 0;
    uint from_offset = 0;
    uint r;
    while (from_offset < a->size) {
        bits_taken += n;
        if (bits_taken > num_bits) {
            bits_taken = 0;
            r = rand();
        }
        if ((r & and_mask) != 0) {
            ++from_offset;
        } else {
            a->indices[to_offset++] = a->indices[from_offset++];
        }
        r >>= n;
    }
    a->size = to_offset;
}

void subsample3(SBA* a, uint n) {
    uint to_offset = 0;
    uint from_offset = 0;
    while (from_offset < a->size) {
        if (rand() % n != 0) {
            ++from_offset;
        } else {
            a->indices[to_offset++] = a->indices[from_offset++];
        }
    }
    a->size = to_offset;
}

void encodeLinear(float input, uint n, SBA* r) {
    uint width = r->size;
    uint start_offset = ceil((n - width) * input);
    for (; width > 0; --width) {
        r->indices[width - 1] = start_offset + width - 1;
    }
}

void encodePeriodic(float input, float period, uint n, SBA* r) {
    float remainder = fmod(input, period);
    uint start_offset = ceil(remainder / period * n);
    int32_t num_wrapped = (int32_t)(start_offset + r->size) - (int32_t)n;
    uint num_remaining;
    if (num_wrapped > 0) {
        for (uint i = 0; i < num_wrapped; ++i) {
            r->indices[i] = i;
        }
        num_remaining = r->size - num_wrapped;
    } else {
        num_remaining = r->size;
        num_wrapped = 0;
    }
    for (uint i = 0; i < num_remaining; ++i) {
        r->indices[i + num_wrapped] = start_offset + i;
    }
}

int main() {
    SBA* a = allocSBA(1);
    // turn_off(a, 0);
    turn_on(a, 1);
    turn_on(a, 2);
    turn_on(a, 0);
    turn_on(a, 4);
    turn_on(a, 4);
    // for (int i = 0; i < 100; ++i) {
    //     print(a);
    //     turn_on(a, rand() % 200);
    //     print(a);
    //     turn_off(a, rand() % 200);
    //     print(a);
    // }
    print(a);
    // SBA* a = allocSBA(8);
    // SBA* b = allocSBA(8);
    // for (int i = 7; i > -1; --i) {
    //     turn_on(a, i);
    //     turn_on(b, i + 1);
    // }
    // SBA* r = a;
    // and(r, a, b);
    // print(r);
    return 0;
}