ROOT_DIR := $(realpath ../../..)
include $(ROOT_DIR)/config.mk

PROJECT := NickNet

SRC_DIR := $(VORTEX_HOME)/tests/opencl/$(PROJECT)

SRCS := $(SRC_DIR)/main.cc

conv1.cl: $(SRC_DIR)/conv1.cl
	cp $< $@

KERNEL_SRCS := conv1.cl

# OPTS ?= -n32

include ../common.mk
