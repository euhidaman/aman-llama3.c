# Compiler settings
CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c11
LDFLAGS = -lm

# Platform specific settings
ifeq ($(OS),Windows_NT)
    TARGET = run.exe
    PLATFORM_SOURCES = win.c
else
    TARGET = run
    PLATFORM_SOURCES =
endif

# Source files
SOURCES = run.c $(PLATFORM_SOURCES)
OBJECTS = $(SOURCES:.c=.o)

# Main targets
all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Object file targets
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean