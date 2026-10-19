CC      = gcc
TARGET  = neural_net

SRC = $(shell find . -name '*.c' -not -path './build/*' | sed 's|^\./||')
INC = $(addprefix -I,$(sort $(dir $(SRC))))

# release default. "make debug" for debug
BUILD     ?= release
BUILD_DIR  = build/$(BUILD)
OBJ        = $(SRC:%.c=$(BUILD_DIR)/%.o)
DEP        = $(OBJ:.o=.d)

ifeq ($(BUILD),debug)
    CFLAGS = -O0 -g -fopenmp -Wall -Wextra -DDEBUG
else
    CFLAGS = -Ofast -fopenmp -Wall
endif
LDFLAGS = -lm

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
