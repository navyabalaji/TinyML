#ifndef TFLM_CONFIG_H
#define TFLM_CONFIG_H

// PSoC-6 specific configurations
#define TFLM_PSOC6 1

// Memory configurations
#define TFLM_ARENA_SIZE (128 * 1024)  // Adjust based on your model size
#define TFLM_TENSOR_ARENA_SIZE (64 * 1024)

// Enable/disable features
#define TF_LITE_STATIC_MEMORY 1
#define TF_LITE_MICRO_ALLOCATIONS_ENABLED 1
#define TF_LITE_MICRO_OPTIMIZED_KERNELS 1

// Debug configurations
#define TF_LITE_MICRO_DEBUG 0
#define TF_LITE_MICRO_DEBUG_DUMP 0

// Include paths
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#endif // TFLM_CONFIG_H 