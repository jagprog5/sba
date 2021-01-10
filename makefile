BUILDDIR = src/py/c-objects
SOURCEDIR = src/c
HEADERDIR = src/c

SOURCES := $(wildcard $(SOURCEDIR)/*.c)
OBJECTS := $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))

CC = gcc
CFLAGS = -std=c99 -Wall -Ofast
BUILDFLAGS = -fPIC # -fPIC can be ommitted for run, but is needed for shared, .so files
LINKFLAGS = -lm
EXECUTABLE := $(BUILDDIR)/exe
SHARED_LIBRARY := $(BUILDDIR)/sba_lib.so

.PHONY: all clean run shared

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $^ -o $@ $(LINKFLAGS)

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c 
	$(CC) -I$(HEADERDIR) -c $< -o $@ $(CFLAGS) $(BUILDFLAGS)

$(SHARED_LIBRARY): $(OBJECTS)
	$(CC) -shared -o $@ $^

clean:
	find $(BUILDDIR) -type f -not -name '.gitignore' -delete -print

run: all
	$(EXECUTABLE)

shared: $(SHARED_LIBRARY)
