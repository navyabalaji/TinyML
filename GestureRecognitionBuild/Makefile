################################################################################
# \file Makefile
# \version 1.0
#
# \brief
# Top-level application make file.
#
################################################################################
# \copyright
# Copyright 2022-2024, Cypress Semiconductor Corporation (an Infineon company)
# SPDX-License-Identifier: Apache-2.0
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

################################################################################
# Basic Configuration
################################################################################

MTB_TYPE=APPLICATION

MTB_PROJECTS=proj_cm0p proj_cm4

include common_app.mk

# TFLM Configuration
TFLM_DIR = $(CY_APP_PATH)/libs/tensorflow
TFLM_INCLUDES = \
    -I$(TFLM_DIR) \
    -I$(TFLM_DIR)/tensorflow/lite/micro \
    -I$(TFLM_DIR)/tensorflow/lite/core \
    -I$(TFLM_DIR)/tensorflow/lite/c \
    -I$(TFLM_DIR)/tensorflow/lite/kernels \
    -I$(TFLM_DIR)/tensorflow/lite/micro/kernels \
    -I$(TFLM_DIR)/tensorflow/lite/micro/memory_planner \
    -I$(TFLM_DIR)/tensorflow/lite/micro/arena_allocator \
    -I$(CY_APP_PATH)/libs/tflm

# Add TFLM flags
CFLAGS += $(TFLM_INCLUDES)
CFLAGS += -DTF_LITE_STATIC_MEMORY
CFLAGS += -DTF_LITE_MICRO_ALLOCATIONS_ENABLED
CFLAGS += -DTF_LITE_MICRO_OPTIMIZED_KERNELS
CFLAGS += -DTF_LITE_DISABLE_X86_NEON
CFLAGS += -DTF_LITE_MCU_DEBUG_LOG
CFLAGS += -DTFLM_PSOC6

# TFLM source files
TFLM_SOURCES = \
    $(CY_APP_PATH)/libs/tflm/tflm_psoc6.c \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_error_reporter.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_interpreter.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/all_ops_resolver.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_utils.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/debug_log.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_allocator.cc \
    $(TFLM_DIR)/tensorflow/lite/micro/micro_profiler.cc

# Add TFLM sources to SOURCES
SOURCES += $(TFLM_SOURCES)

include $(CY_TOOLS_DIR)/make/application.mk
