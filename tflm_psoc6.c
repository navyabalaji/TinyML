#include "tflm_psoc6.h"
#include "cy_pdl.h"
#include "tensorflow/lite/error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_profiler.h"

// PSoC 6 specific optimizations
static void tflm_psoc6_optimize(void) {
    // Cache optimization
    SCB_EnableICache();
    SCB_EnableDCache();
    
    // Memory alignment optimization
    // Note: PSoC 6 has 32-bit memory bus
    // Ensure tensor arena is 32-bit aligned
}

cy_rslt_t tflm_init(tflm_context_t* context, const uint8_t* model_data) {
    if (context == NULL || model_data == NULL) {
        return CY_RSLT_TYPE_ERROR;
    }

    // Apply PSoC 6 specific optimizations
    tflm_psoc6_optimize();

    // Initialize error reporter
    context->error_reporter = &context->micro_error_reporter;

    // Get model from flatbuffer
    context->model = tflite::GetModel(model_data);
    if (context->model->version() != TFLITE_SCHEMA_VERSION) {
        context->error_reporter->Report("Model schema mismatch!");
        return CY_RSLT_TYPE_ERROR;
    }

    // Create tensor arena with 32-bit alignment
    context->tensor_arena = (uint8_t*)aligned_alloc(4, TFLM_TENSOR_ARENA_SIZE);
    if (context->tensor_arena == NULL) {
        context->error_reporter->Report("Tensor arena allocation failed");
        return CY_RSLT_TYPE_ERROR;
    }
    context->tensor_arena_size = TFLM_TENSOR_ARENA_SIZE;

    // Create operations resolver using AllOpsResolver
    static tflite::AllOpsResolver resolver;

    // Create micro allocator
    static tflite::MicroAllocator static_allocator(
        context->tensor_arena, context->tensor_arena_size, context->error_reporter);
    context->allocator = &static_allocator;

    // Create profiler
    static tflite::MicroProfiler static_profiler;
    context->profiler = &static_profiler;

    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        context->model, resolver, context->allocator, context->error_reporter,
        context->profiler);
    context->interpreter = &static_interpreter;

    // Allocate tensors
    if (context->interpreter->AllocateTensors() != kTfLiteOk) {
        context->error_reporter->Report("Tensor allocation failed");
        return CY_RSLT_TYPE_ERROR;
    }

    return CY_RSLT_SUCCESS;
}

cy_rslt_t tflm_invoke(tflm_context_t* context) {
    if (context == NULL || context->interpreter == NULL) {
        return CY_RSLT_TYPE_ERROR;
    }

    if (context->interpreter->Invoke() != kTfLiteOk) {
        context->error_reporter->Report("Invoke failed");
        return CY_RSLT_TYPE_ERROR;
    }

    return CY_RSLT_SUCCESS;
}

TfLiteTensor* tflm_get_input_tensor(tflm_context_t* context, int index) {
    if (context == NULL || context->interpreter == NULL) {
        return NULL;
    }
    if (index < 0 || index >= context->interpreter->inputs_size()) {
        context->error_reporter->Report("Invalid input tensor index");
        return NULL;
    }
    return context->interpreter->input(index);
}

TfLiteTensor* tflm_get_output_tensor(tflm_context_t* context, int index) {
    if (context == NULL || context->interpreter == NULL) {
        return NULL;
    }
    if (index < 0 || index >= context->interpreter->outputs_size()) {
        context->error_reporter->Report("Invalid output tensor index");
        return NULL;
    }
    return context->interpreter->output(index);
}

void tflm_deinit(tflm_context_t* context) {
    if (context != NULL) {
        if (context->tensor_arena != NULL) {
            free(context->tensor_arena);
            context->tensor_arena = NULL;
        }
        context->interpreter = NULL;
        context->model = NULL;
        context->error_reporter = NULL;
        context->allocator = NULL;
        context->profiler = NULL;
        
        // Disable cache if needed
        // SCB_DisableICache();
        // SCB_DisableDCache();
    }
} 