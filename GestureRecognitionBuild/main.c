#include "main.h"

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
void initialize_hardware(void) {
    /* Initialize system */
    cy_rslt_t result;
    
    /* Initialize the device and board peripherals */
    result = cybsp_init();
    CY_ASSERT(result == CY_RSLT_SUCCESS);
    
    /* Enable global interrupts */
    __enable_irq();
    
    /* Initialize retarget-io for UART output */
    cy_retarget_io_init(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX, CY_RETARGET_IO_BAUDRATE);
    
    /* Initialize camera interface */
    setup_camera();
}

/* Function to setup TensorFlow Lite Micro */
bool setup_tensorflow(void) {
    /* Set up logging */
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;
    
    /* Map the model into a usable data structure */
    model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        TF_LITE_REPORT_ERROR(error_reporter,
            "Model provided is schema version %d not equal "
            "to supported version %d.",
            model->version(), TFLITE_SCHEMA_VERSION);
        return false;
    }
    
    /* Pull in only the operation implementations we need */
    static tflite::AllOpsResolver resolver;
    
    /* Build an interpreter to run the model with */
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;
    
    /* Allocate memory from the tensor_arena for the model's tensors */
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
        return false;
    }
    
    /* Get pointers to the model's input and output tensors */
    input = interpreter->input(0);
    output = interpreter->output(0);
    
    return true;
}

/* Function to process gesture */
void process_gesture(uint8_t* image_data) {
    /* Copy image data to input tensor */
    for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
        input->data.int8[i] = (int8_t)(image_data[i] - 128);  // Convert to signed int8
    }
    
    /* Run inference */
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        printf("Invoke failed\n");
        return;
    }
    
    /* Find the index with highest probability */
    int8_t max_score = output->data.int8[0];
    int max_index = 0;
    for (int i = 1; i < NUM_CLASSES; i++) {
        if (output->data.int8[i] > max_score) {
            max_score = output->data.int8[i];
            max_index = i;
        }
    }
    
    /* Print the result */
    printf("Detected gesture: %s (confidence: %d)\n", 
           gesture_labels[max_index], max_score);
}

/* Main function */
int main(void) {
    /* Initialize hardware */
    initialize_hardware();
    printf("Hardware initialized\n");
    
    /* Setup TensorFlow Lite Micro */
    if (!setup_tensorflow()) {
        printf("TensorFlow setup failed\n");
        return -1;
    }
    printf("TensorFlow initialized\n");
    
    /* Buffer for image data */
    uint8_t image_buffer[IMAGE_SIZE * IMAGE_SIZE];
    
    /* Main loop */
    for (;;) {
        /* Capture image */
        capture_image(image_buffer);
        
        /* Process the gesture */
        process_gesture(image_buffer);
        
        /* Wait before next capture */
        cyhal_system_delay_ms(1000);
    }
} 