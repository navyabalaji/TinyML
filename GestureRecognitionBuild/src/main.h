#ifndef MAIN_H
#define MAIN_H

#include "cy_pdl.h"
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"

/* TensorFlow Lite for Microcontrollers */
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

/* Model data */
extern const unsigned char model_tflite[];
extern const int model_tflite_len;

/* Constants */
#define IMAGE_SIZE 28
#define NUM_CHANNELS 1
#define NUM_CLASSES 24
#define TENSOR_ARENA_SIZE (96 * 1024)

/* Function declarations */
void initialize_hardware(void);
void process_gesture(uint8_t* image_data);
void setup_camera(void);
void capture_image(uint8_t* buffer);

#endif /* MAIN_H */ 