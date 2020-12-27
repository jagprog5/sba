#include "sba.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

SBA* allocSBA(uint inital_cap) {
    SBA* a = malloc(sizeof(*a));
    a->indices = malloc(sizeof(uint) * inital_cap);
    a->capacity = inital_cap;
    a->size = 0;
    return a;
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

void insert(SBA* a, uint bit_index) {
    if (a->size >= a->capacity) {
        a->capacity <<= 1;
        if (a->capacity == 0) {
            a->capacity = 1;
        }
        a->indices = realloc(a->indices, sizeof(uint) * a->capacity);
    }
    if (a->size == 0) {
        // prevent undefined behaviour when inserting into empty list
        a->indices[0] = 0;
    }

    uint left = 0;
    uint right = a->size;
    uint middle;

    binsearch:
    middle = (right + left) / 2;
    if (a->indices[middle] > bit_index) {
        right = middle;
    } else {
        left = middle;
    }
    if (right - left > 1) {
        goto binsearch;
    }

    if (right == middle && left != middle) {
        // L  M  R
        // L  MR        "right = middle;"
        // LM R         "middle -= 1;""
        middle -= 1;
    }
    
    if (a->size != 0) {
        if (a->indices[middle] == bit_index) {
            return; // skip duplicates
        } else if (a->indices[middle] < bit_index) {
            // insert on right side of item if needed
            middle += 1;
        }
    }

    a->size += 1;
    memmove(a->indices + middle + 1, a->indices + middle, sizeof(uint) * (a->size - middle));
    a->indices[middle] = bit_index;
}

SBA* and(SBA* a, SBA* b) {
    SBA* o = allocSBA(a->size < b->size ? a->size : b->size);
    uint a_offset = 0;
    uint a_val;
    uint b_offset = 0;
    uint b_val;

    get_both:
    if (a_offset >= a->size) {
        goto end;
    }
    a_val = a->indices[a_offset++];
    if (b_offset >= b->size) {
        goto end;
    }
    b_val = b->indices[b_offset++];

    loop:
    if (a_val < b_val) {
        // get a
        if (a_offset >= a->size) {
            goto end;
        }
        a_val = a->indices[a_offset++];
        goto loop;
    } else if (a_val == b_val) {
        o->indices[o->size++] = a_val;
        goto get_both;
    } else {
        // get b
        if (b_offset >= b->size) {
            goto end;
        }
        b_val = b->indices[b_offset++];
        goto loop;
    }

    end:
    return o;
}

SBA* or(SBA* a, SBA* b) {
    SBA* u = allocSBA(a->size + b->size);
    uint a_offset = 0;
    uint a_val;
    uint b_offset = 0;
    uint b_val;

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
        u->indices[u->size++] = a_val;
        goto get_a;
    } else if ((a && b && a_val > b_val) || (!a && b)) {
        u->indices[u->size++] = b_val;
        goto get_b;
    } else if (a && b && a_val == b_val) {
        u->indices[u->size++] = a_val;
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
    return u;
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
    SBA* a = allocSBA(4);
    a->size = a->capacity;
    for (float i = 0; i <=1; i += 0.1f) {
        printf("%.2f: ", i);
        encodePeriodic(i, 0.5f, 15, a);
        print(a);
    }
    free(a);
    // srand((uint)time(NULL));
    SBA* a = allocSBA(0);
    insert(a, 5);
    insert(a, 5);
    insert(a, 1);
    insert(a, 2);
    insert(a, 7);
    SBA* b = allocSBA(0);
    insert(b, 0);
    insert(b, 2);
    insert(b, 7);
    insert(b, 8);
    insert(b, 9);
    SBA* o = or(a,b);
    // SBA* s = subsample(o, 4);
    // // printf("%u\n", s->size);
    print(o);
    // print(s);
    // free(a);
    // free(b);
    // free(o);
    // free(s);
    return 0;
}