#ifndef MAIN_H
#define MAIN_H

#include "cy_pdl.h"
#include "tflm_psoc6.h"

// Image dimensions
#define IMAGE_WIDTH 28
#define IMAGE_HEIGHT 28
#define IMAGE_CHANNELS 1

// Number of gestures
#define NUM_GESTURES 24

// Gesture labels
extern const char* gesture_labels[NUM_GESTURES];

// Model data
extern const unsigned char model_tflite[];
extern const int model_tflite_len;

// Function declarations
cy_rslt_t initialize_hardware(void);
cy_rslt_t initialize_tflm(void);
cy_rslt_t process_gesture(const uint8_t* image_data);
void print_gesture_result(int gesture_index, float confidence);

#endif /* MAIN_H */ 