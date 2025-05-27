#ifndef TFLM_PSOC6_H
#define TFLM_PSOC6_H

#include "cy_pdl.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_profiler.h"

// Tensor arena size (96KB)
#define TFLM_TENSOR_ARENA_SIZE (96 * 1024)

// TFLM context structure
typedef struct {
    tflite::MicroErrorReporter micro_error_reporter;
    tflite::ErrorReporter* error_reporter;
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    tflite::MicroAllocator* allocator;
    tflite::MicroProfiler* profiler;
    uint8_t* tensor_arena;
    size_t tensor_arena_size;
} tflm_context_t;

// Function declarations
cy_rslt_t tflm_init(tflm_context_t* context, const uint8_t* model_data);
cy_rslt_t tflm_invoke(tflm_context_t* context);
TfLiteTensor* tflm_get_input_tensor(tflm_context_t* context, int index);
TfLiteTensor* tflm_get_output_tensor(tflm_context_t* context, int index);
void tflm_deinit(tflm_context_t* context);

#endif /* TFLM_PSOC6_H */ 