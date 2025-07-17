#ifndef ML_MODEL_H
#define ML_MODEL_H

typedef enum {
    LAYER_DENSE,
    LAYER_RELU,
    LAYER_SOFTMAX,
    LAYER_UNKNOWN
} LayerType;

typedef struct {
    int in_dim;
    int out_dim;
    const float* weights;
    const float* biases;
} DenseLayer;

typedef struct {
    LayerType type;
    union {
        DenseLayer dense;
        // Add other layer types if needed
    };
} Layer;

typedef struct {
    int num_layers;
    const Layer* layers;
} NeuralNet;

#endif // ML_MODEL_H 