BUILDDIR = build
SOURCEDIR = src
HEADERDIR = headers

SOURCES := $(wildcard $(SOURCEDIR)/*.c)
OBJECTS := $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))
SHARED_LIBRARIES := $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.so,$(SOURCES))

CC = gcc
CFLAGS = -std=c99 -Wall # -Ofast 
LINKFLAGS = -lm
EXECUTABLE := $(BUILDDIR)/exe

.PHONY: all clean run shared

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $^ -o $@ $(LINKFLAGS)

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c # -fPIC can be ommitted for run, but is needed for shared, .so files
	$(CC) -I$(HEADERDIR) -c $< -o $@ $(CFLAGS) -fPIC

$(BUILDDIR)/%.so: $(BUILDDIR)/%.o
	$(CC) -shared -fPIC -o $@ $<

clean:
	find $(BUILDDIR) -type f -not -name '.gitignore' -delete -print

run: all
	$(EXECUTABLE)

shared: $(SHARED_LIBRARIES)
