#include "main.h"
#include "cy_pdl.h"
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"

/* Model data */
extern const unsigned char model_tflite[];
extern const int model_tflite_len;

/* TFLM context */
static tflm_context_t tflm_context;

/* TensorFlow Lite for Microcontrollers */
namespace {
    tflite::ErrorReporter* error_reporter = nullptr;
    const tflite::Model* model = nullptr;
    tflite::MicroInterpreter* interpreter = nullptr;
    TfLiteTensor* input = nullptr;
    TfLiteTensor* output = nullptr;
    uint8_t tensor_arena[TENSOR_ARENA_SIZE];
}

/* Gesture labels */
const char* gesture_labels[] = {
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "Y"  // Note: X and Z are not included as they require motion
};

/* Function to initialize hardware */
cy_rslt_t initialize_hardware(void) {
    cy_rslt_t result;

    /* Initialize the device and board peripherals */
    result = cybsp_init();
    if (result != CY_RSLT_SUCCESS) {
        return result;
    }

    /* Initialize retarget-io to use the debug UART port */
    cy_retarget_io_init(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX, CY_RETARGET_IO_BAUDRATE);

    /* Initialize the camera */
    // TODO: Add camera initialization code

    return CY_RSLT_SUCCESS;
}

/* Function to setup TensorFlow Lite Micro */
cy_rslt_t initialize_tflm(void) {
    /* Initialize TFLM */
    return tflm_init(&tflm_context, model_tflite);
}

/* Function to process gesture */
cy_rslt_t process_gesture(const uint8_t* image_data) {
    cy_rslt_t result;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    float max_value = 0.0f;
    int max_index = -1;

    /* Get input tensor */
    input_tensor = tflm_get_input_tensor(&tflm_context, 0);
    if (input_tensor == NULL) {
        return CY_RSLT_TYPE_ERROR;
    }

    /* Copy image data to input tensor */
    memcpy(input_tensor->data.data, image_data, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS);

    /* Run inference */
    result = tflm_invoke(&tflm_context);
    if (result != CY_RSLT_SUCCESS) {
        return result;
    }

    /* Get output tensor */
    output_tensor = tflm_get_output_tensor(&tflm_context, 0);
    if (output_tensor == NULL) {
        return CY_RSLT_TYPE_ERROR;
    }

    /* Find the gesture with highest confidence */
    for (int i = 0; i < NUM_GESTURES; i++) {
        float value = output_tensor->data.f[i];
        if (value > max_value) {
            max_value = value;
            max_index = i;
        }
    }

    /* Print result */
    print_gesture_result(max_index, max_value);

    return CY_RSLT_SUCCESS;
}

void print_gesture_result(int gesture_index, float confidence) {
    printf("Detected gesture: %s with confidence: %.2f%%\r\n", 
           gesture_labels[gesture_index], confidence * 100.0f);
}

/* Main function */
int main(void) {
    cy_rslt_t result;
    uint8_t image_buffer[IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS];

    /* Initialize hardware */
    result = initialize_hardware();
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    /* Initialize TFLM */
    result = initialize_tflm();
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    printf("Gesture recognition system initialized\r\n");

    /* Main loop */
    for (;;) {
        /* Capture image */
        // TODO: Add camera capture code
        // capture_image(image_buffer);

        /* Process gesture */
        result = process_gesture(image_buffer);
        if (result != CY_RSLT_SUCCESS) {
            printf("Error processing gesture\r\n");
        }

        /* Wait for next frame */
        cyhal_system_delay_ms(100);
    }
} 