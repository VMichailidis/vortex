ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := sparse

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

kernel.cl: $(SRC_DIR)/kernel.cl
	cp $< $@

KERNEL_SRCS := kernel.cl

# OPTS ?= -n32
CXXFLAGS += -fdiagnostics-color=always -Wno-format -g
VALGRIND_OPTS = --track-origins=yes
debug:
	LD_LIBRARY_PATH=$(POCL_PATH)/lib:$(VORTEX_RT_PATH):$(LLVM_VORTEX)/lib:$(LD_LIBRARY_PATH) $(POCL_CC_FLAGS) VORTEX_DRIVER=simx valgrind $(VALGRIND_OPTS) ./$(PROJECT) $(OPTS)

include ../common.mk
