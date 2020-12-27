BUILDDIR = build
SOURCEDIR = src
HEADERDIR = headers

SOURCES := $(wildcard $(SOURCEDIR)/*.c)
OBJECTS := $(patsubst $(SOURCEDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))

CC = gcc
CFLAGS = -std=c99 -Wall # -Ofast 
LINKFLAGS = -lm # -lncurses
EXECUTABLE := $(BUILDDIR)/exe

.PHONY: all clean run

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $^ -o $@ $(LINKFLAGS)

$(BUILDDIR)/%.o: $(SOURCEDIR)/%.c
	$(CC) -I$(HEADERDIR) -c $< -o $@ $(CFLAGS)

clean:
	rm -vf $(EXECUTABLE) $(OBJECTS)

run: all
	$(EXECUTABLE)
