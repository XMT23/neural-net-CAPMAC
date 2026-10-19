CC      = gcc
TARGET  = neural_net

SRC = $(shell find . -name '*.c' -not -path './build/*' | sed 's|^\./||')
INC = $(addprefix -I,$(sort $(dir $(SRC))))

# release default. "make debug" for debug
PAR				?= 1
BUILD     ?= release
OBJ        = $(SRC:%.c=$(BUILD_DIR)/%.o)
DEP        = $(OBJ:.o=.d)

ifeq ($(BUILD),debug)
    CFLAGS = -O0 -g -Wall -Wextra -DDEBUG
else
    CFLAGS = -Ofast -Wall
endif

ifeq ($(PAR),1)
	CFLAGS 	+= -fopenmp
	SUFFIX	 = par
else
	CFLAGS 	+= -Wno-unknown-pragmas
	SUFFIX	 = seq
endif

LDFLAGS = -lm

BUILD_DIR  = build/$(BUILD)-$(SUFFIX)

.PHONY: all debug clean


all: $(BUILD_DIR)/$(TARGET)

debug:
	$(MAKE) BUILD=debug

$(BUILD_DIR)/$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(DEP)

clean:
	rm -rf build
