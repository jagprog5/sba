#include "sba.h"

#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef _WIN32
void PyInit_sba_lib() {}
#endif

SBA* _allocSBA_nosetsize(uint32_t initialCap) {
    SBA* a = malloc(sizeof(*a));
    a->indices = malloc(sizeof(*a->indices) * initialCap);
    a->capacity = initialCap;
    return a;
}

SBA* allocSBA(uint32_t initialCap) {
    SBA* a = _allocSBA_nosetsize(initialCap);
    a->size = 0;
    return a;
}

void freeSBA(SBA* a) {
    free(a->indices);
    free(a);
}

void shortenSBA(SBA* a) {
    a->capacity = a->size;
    a->indices = realloc(a->indices, sizeof(*a->indices) * a->capacity);
}

void printSBA(SBA* a) {
    if (a->size != 0) {
        uint_fast32_t amount = 0;
        for (uint_fast32_t i = 0; i < a->size - 1; ++i) {
            uint_fast32_t val = a->indices[i];
            amount += 2 + (val == 0 ? 0 : floorf(log10f(val)));
        }
        for (uint_fast32_t j = 0; j < amount; ++j) {
            putchar(' ');
        }
        printf("V\n");
    }
    for (uint_fast32_t i = 0; i < a->capacity; ++i) {
        printf("%d ", a->indices[i]);
    }
    putchar('\n');
}

void turnOn(SBA* a, uint32_t bitIndex) {
    int_fast32_t left = 0;
    int_fast32_t right = (int_fast32_t)a->size - 1;
    int_fast32_t middle = 0;
    uint_fast32_t mid_val = 0;
    while (left <= right) {
        middle = (right + left) / 2;
        mid_val = a->indices[middle];
        if (mid_val > bitIndex) {
            left = middle + 1;
        } else if (mid_val < bitIndex) {
            right = middle - 1;
        } else {
            return; // skip duplicate
        }
    }
    if (bitIndex < mid_val) {
        middle += 1;
    }
    if (a->size >= a->capacity) {
        a->capacity = a->capacity + (a->capacity >> 1) + 1; // cap *= 1.5 + 1, estimate for golden ratio
        a->indices = realloc(a->indices, sizeof(*a->indices) * a->capacity);
    }
    memmove(a->indices + middle + 1, a->indices + middle, sizeof(*a->indices) * (a->size - middle));
    a->size += 1;
    a->indices[middle] = bitIndex;
}

void turnOff(SBA* a, uint32_t bitIndex) {
    int_fast32_t left = 0;
    int_fast32_t right = (int_fast32_t)a->size - 1;
    int_fast32_t middle;
    while (left <= right) {
        middle = (right + left) / 2;
        uint_fast32_t mid_val = a->indices[middle];
        if (mid_val == bitIndex) {
            a->size -= 1;
            memmove(a->indices + middle, a->indices + middle + 1, sizeof(*a->indices) * (a->size - middle));
            if (a->size < a->capacity >> 1) {
                shortenSBA(a);
            }
            return;
        } else if (mid_val > bitIndex) {
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
}

uint8_t getBit(SBA* a, uint32_t bitIndex) {
    int_fast32_t left = 0;
    int_fast32_t right = (int_fast32_t)a->size - 1;
    int_fast32_t middle;
    while (left <= right) {
        middle = (right + left) / 2;
        uint_fast32_t mid_val = a->indices[middle];
        if (mid_val == bitIndex) {
            return 1;
        } else if (mid_val > bitIndex) {
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
    return 0;
}

SBA* allocSBA_getSection(uint32_t stop_inclusive, uint32_t start_inclusive) {
    return _allocSBA_nosetsize(stop_inclusive - start_inclusive + 1);
}

void getSection(SBA* r, SBA* in, uint32_t stop_inclusive, uint32_t start_inclusive) {
    uint_fast32_t r_size = 0;
    int_fast32_t left = 0;
    int_fast32_t right = (int_fast32_t)in->size - 1;
    int_fast32_t middle;
    uint_fast32_t mid_val;
    while (left <= right) {
        middle = (right + left) / 2;
        mid_val = in->indices[middle];
        if (mid_val == stop_inclusive) {
            break;
        } else if (mid_val > stop_inclusive) {
            left = middle + 1;
        } else {
            right = middle - 1;
        }
    }
    if (stop_inclusive < mid_val) {
        middle += 1;
        mid_val = in->indices[middle];
    }

    uint_fast32_t in_size = in->size;
    while (mid_val >= start_inclusive) {
        r->indices[r_size++] = mid_val;
        middle += 1;
        if (middle >= in_size) {
            break; // ran off end
        }
        mid_val = in->indices[middle];
    }
    r->size = r_size;
}

void turnOffAll(SBA* a, SBA* rm) {
    uint_fast32_t a_size = a->size;
    uint_fast32_t rm_size = rm->size;
    uint_fast32_t a_from = 0;
    uint_fast32_t a_to = 0;
    uint_fast32_t a_val;
    uint_fast32_t rm_offset = 0;
    uint_fast32_t rm_val;
    while (a_from < a_size) {
        if (rm_offset < rm_size) {
            a_val = a->indices[a_from];
            rm_val = rm->indices[rm_offset];
            if (rm_val > a_val) {
                rm_offset += 1;
                continue;
            } else if (rm_val == a_val) {
                rm_offset += 1;
                a_from += 1;
                continue;
            }
        }
        a->indices[a_to++] = a->indices[a_from++];
    }
    a->size = a_to;
}

SBA* allocSBA_andBits(SBA* a, SBA* b) {
    return _allocSBA_nosetsize(a->size < b->size ? a->size : b->size); // and sets the size
}

void andBits(void* r, SBA* a, SBA* b, uint8_t size_only) {
    uint_fast32_t a_offset = 0;
    uint_fast32_t a_val;
    uint_fast32_t a_size = a->size; // store in case r = a
    uint_fast32_t b_offset = 0;
    uint_fast32_t b_val;
    uint_fast32_t b_size = b->size; // store in case r = b
    uint_fast32_t r_size = 0;
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
    if (a_val > b_val) {
        // get a
        if (a_offset >= a_size) {
            goto end;
        }
        a_val = a->indices[a_offset++];
        goto loop;
    } else if (a_val == b_val) {
        if (!size_only) {
            ((SBA*)r)->indices[r_size] = a_val;
        }
        r_size += 1;
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
    if (size_only) {
        *((uint32_t*)r) = r_size;
    } else {
        ((SBA*)r)->size = r_size;
    }
}

SBA* allocSBA_or(SBA* a, SBA* b) {
    return _allocSBA_nosetsize(a->size + b->size);
}

void orBits(void* r, SBA* a, SBA* b, uint8_t exclusive, uint8_t size_only) {
    uint_fast32_t a_offset = 0;
    uint_fast32_t a_val;
    uint_fast32_t b_offset = 0;
    uint_fast32_t b_val;
    uint_fast32_t r_size = 0;

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
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    if ((a && b && a_val > b_val) || (a && !b)) {
        if (!size_only) {
            ((SBA*)r)->indices[r_size] = a_val;
        }
        r_size += 1;
        goto get_a;
    } else if ((a && b && a_val < b_val) || (!a && b)) {
        if (!size_only) {
            ((SBA*)r)->indices[r_size] = b_val;
        }
        r_size += 1;
        goto get_b;
    } else if (a && b && a_val == b_val) {
        if (!exclusive) {
            if (!size_only) {
                ((SBA*)r)->indices[r_size] = a_val;
            }
            r_size += 1;
        }
        goto get_both;
    }
    #pragma GCC diagnostic pop

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
    if (size_only) {
        *((uint32_t*)r) = r_size;
    } else {
        ((SBA*)r)->size = r_size;
    }
}

void rshift(SBA* a, uint32_t n) {
    uint_fast32_t a_size = a->size;
    uint_fast32_t index = 0;
    while (index < a_size) {
        uint_fast32_t val = a->indices[index];
        if (val < n) {
            a->size = index;
            return;
        }
        a->indices[index] = val - n;
        ++index;
    }
}

void lshift(SBA* a, uint32_t n) {
    for (uint_fast32_t i = 0; i < a->size; ++i) {
        a->indices[i] += n;
    }
}

uint8_t equal(SBA* a, SBA* b) {
    if (a->size != b->size) {
        return 0;
    }
    for (uint32_t i = 0; i < a->size; ++i) {
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
    memcpy(dest->indices, src->indices, sizeof(*dest->indices) * dest->size);
}

void subsample(SBA* a, float amount) {
    unsigned int check_val = amount * RAND_MAX;
    uint_fast32_t to_offset = 0;
    uint_fast32_t from_offset = 0;
    while (from_offset < a->size) {
        if ((unsigned int)rand() > check_val) {
            ++from_offset;
        } else {
            a->indices[to_offset++] = a->indices[from_offset++];
        }
    }
    a->size = to_offset;
}

void encodeLinear(float input, uint32_t n, SBA* r) {
    uint_fast32_t width = r->capacity;
    uint_fast32_t start_offset = roundf((n - width) * input);
    for (uint_fast32_t i = 0; i < width; ++i) {
        r->indices[i] = start_offset + width - i - 1;
    }
}

void encodePeriodic(float input, float period, uint32_t n, SBA* r) {
    uint_fast32_t cap = r->capacity;
    float progress = input / period;
    progress = progress - (int)progress;
    uint_fast32_t start_offset = roundf(progress * n);
    if (start_offset + cap > n) {
        uint_fast32_t num_wrapped = start_offset + cap - n;
        uint_fast32_t num_leading = n - start_offset;
        do {
            --num_leading;
            r->indices[num_leading] = n - num_leading - 1;
        } while (num_leading > 0);
        do {
            --num_wrapped;
            r->indices[cap - num_wrapped - 1] = num_wrapped;
        } while (num_wrapped > 0);
    } else {
        for (uint_fast32_t i = 0; i < cap; ++i) {
            r->indices[i] = cap - i - 1 + start_offset;
        }
    }
}

void seed_rand() {
    srand(time(NULL));
}
