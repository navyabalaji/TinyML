# Import required libraries
import tensorflow as tf  # Deep learning framework
from sklearn.datasets import load_iris  # IRIS dataset loader
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.preprocessing import StandardScaler  # For feature standardization
from sklearn.metrics import accuracy_score  # For calculating model accuracy
import numpy as np  # For numerical operations

def create_model(n_hidden_layers, neurons_per_layer, input_shape, num_classes):
    """
    Creates a DNN model with specified architecture.
    
    Args:
        n_hidden_layers (int): Number of hidden layers in the network
        neurons_per_layer (int): Number of neurons in each hidden layer
        input_shape (tuple): Shape of input data
        num_classes (int): Number of output classes
    
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    # Create a sequential model (linear stack of layers)
    model = tf.keras.Sequential()
    
    # Add input layer with specified shape
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # Add hidden layers with ReLU activation
    for _ in range(n_hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    
    # Add output layer with softmax activation for multi-class classification
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Compile the model with Adam optimizer and appropriate loss function
    model.compile(optimizer='adam', 
                 loss='sparse_categorical_crossentropy',  # Suitable for integer labels
                 metrics=['accuracy'])
    return model


if __name__ == '__main__':
    # Load the IRIS dataset
    iris = load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels

    # Split data into training (80%) and testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features to have zero mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)  # Fit and transform training data
    X_test = scaler.transform(X_test)  # Transform test data using training statistics

    # Define input shape and number of classes
    input_shape = (X_train.shape[1],)  # Number of features
    num_classes = len(np.unique(y))  # Number of unique classes in target

    # Define hyperparameters for grid search
    hidden_layers_options = [1, 2, 3]  # Test different numbers of hidden layers
    neurons_options = [8, 16, 32, 64]  # Test different numbers of neurons per layer

    # Initialize variables to track best model
    best_accuracy = 0
    best_params = {}
    best_model = None

    # Perform grid search over hyperparameters
    for n_layers in hidden_layers_options:
        for n_neurons in neurons_options:
            print(f"\nTraining with {n_layers} layers and {n_neurons} neurons per layer")
            
            # Create and compile model with current hyperparameters
            model = create_model(n_layers, n_neurons, input_shape, num_classes)
            
            # Calculate and print number of parameters in the model
            model_parameters = model.count_params()
            print(f"Model parameters: {model_parameters}")

            # Train the model
            history = model.fit(X_train, y_train, 
                              epochs=50,  # Number of training iterations
                              batch_size=16,  # Number of samples per gradient update
                              verbose=0)  # Suppress training progress output
            
            # Evaluate model on test set
            _, accuracy = model.evaluate(X_test, y_test, verbose=0)
            print(f"Accuracy: {accuracy:.4f}")

            # Check if model meets accuracy constraint and has fewer parameters
            if accuracy > 0.80:  # Constraint: Accuracy > 80%
                if best_model is None or model_parameters < best_model.count_params():
                    best_accuracy = accuracy
                    best_params = {'n_layers': n_layers, 'n_neurons': n_neurons}
                    best_model = model
                    print(f"Found a better model with {model_parameters} parameters and accuracy {accuracy:.4f}")

    if best_model:
        print(f"\nBest model found with parameters: {best_params} and accuracy: {best_accuracy:.4f}")
        print("\nQuantizing the best model to int8...")

        # Initialize TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable default optimizations

        # Define representative dataset generator for quantization
        def representative_data_gen():
            """
            Generator function that yields representative data for quantization.
            Uses training data to determine quantization parameters.
            """
            for input_value in X_train.astype(np.float32):
                yield [input_value.reshape(1, -1)]  # Reshape to match model input shape

        # Set up quantization parameters
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        try:
            # Convert and save the quantized model
            quantized_tflite_model = converter.convert()
            with open('iris_dnn_quantized.tflite', 'wb') as f:
                f.write(quantized_tflite_model)
            print("Quantized model saved as 'iris_dnn_quantized.tflite'")

            # Set up TFLite interpreter for model verification
            interpreter = tf.lite.Interpreter(model_content=quantized_tflite_model)
            interpreter.allocate_tensors()

            # Get input and output tensor details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Prepare test data for quantized model
            input_scale, input_zero_point = input_details[0]['quantization']
            # Quantize test data using the same scale and zero point
            X_test_quantized = (X_test / input_scale + input_zero_point).astype(input_details[0]['dtype'])
            
            # Evaluate quantized model on test set
            quantized_predictions = []
            for i in range(len(X_test_quantized)):
                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], X_test_quantized[i:i+1])
                # Run inference
                interpreter.invoke()
                # Get output tensor
                output = interpreter.get_tensor(output_details[0]['index'])
                # Store prediction
                quantized_predictions.append(np.argmax(output[0]))

            # Calculate and print accuracy of quantized model
            quantized_accuracy = accuracy_score(y_test, quantized_predictions)
            print(f"Quantized model accuracy (on test set): {quantized_accuracy:.4f}")

        except Exception as e:
            print(f"Error during quantization or verification: {e}")

    else:
        print("No model found that meets the accuracy constraint (>80%). Consider adjusting hyperparameters or epochs.")
