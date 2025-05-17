# import os
# import numpy as np
# import tensorflow as tf
# import kagglehub
# from tensorflow.keras import layers, models
# from sklearn.model_selection import train_test_split
# import pandas as pd

# # Set random seed for reproducibility
# tf.random.set_seed(42)
# np.random.seed(42)

# def load_and_preprocess_data():
#     # Download the dataset
#     path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    
#     # Load training data
#     train_data = pd.read_csv(os.path.join(path, "sign_mnist_train.csv"))
#     test_data = pd.read_csv(os.path.join(path, "sign_mnist_test.csv"))
    
#     # Separate features and labels
#     y_train = train_data['label']
#     X_train = train_data.drop('label', axis=1)
#     y_test = test_data['label']
#     X_test = test_data.drop('label', axis=1)
    
#     # Reshape and normalize the data
#     X_train = X_train.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
#     X_test = X_test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
#     # Convert labels to categorical
#     y_train = tf.keras.utils.to_categorical(y_train, num_classes=24)
#     y_test = tf.keras.utils.to_categorical(y_test, num_classes=24)
    
#     return X_train, y_train, X_test, y_test

# def create_model():
#     model = models.Sequential([
#         # First Convolutional Layer
#         layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
        
#         # Second Convolutional Layer
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.BatchNormalization(),
#         layers.MaxPooling2D((2, 2)),
        
#         # Third Convolutional Layer
#         layers.Conv2D(64, (3, 3), activation='relu'),
#         layers.BatchNormalization(),
        
#         # Flatten and Dense Layers
#         layers.Flatten(),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.5),
#         layers.Dense(24, activation='softmax')
#     ])
    
#     return model

# def train_model(model, X_train, y_train, X_test, y_test):
#     # Compile the model
#     model.compile(optimizer='adam',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
    
#     # Add callbacks
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.2,
#             patience=3
#         )
#     ]
    
#     # Train the model
#     history = model.fit(
#         X_train, y_train,
#         epochs=50,
#         batch_size=32,
#         validation_data=(X_test, y_test),
#         callbacks=callbacks
#     )
    
#     return history

# def convert_to_tflite(model):
#     # Convert to TFLite model
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
#     # Enable quantization
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#     converter.inference_input_type = tf.int8
#     converter.inference_output_type = tf.int8
    
#     # Representative dataset generator
#     def representative_dataset_gen():
#         for i in range(100):
#             yield [X_train[i:i+1]]
    
#     converter.representative_dataset = representative_dataset_gen
    
#     # Convert the model
#     tflite_model = converter.convert()
    
#     # Save the model
#     with open('../src/model.tflite', 'wb') as f:
#         f.write(tflite_model)

# def main():
#     # Create necessary directories
#     os.makedirs('../src', exist_ok=True)
    
#     # Load and preprocess data
#     print("Loading and preprocessing data...")
#     X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
#     # Create and train model
#     print("Creating and training model...")
#     model = create_model()
#     history = train_model(model, X_train, y_train, X_test, y_test)
    
#     # Evaluate model
#     print("Evaluating model...")
#     test_loss, test_accuracy = model.evaluate(X_test, y_test)
#     print(f"Test accuracy: {test_accuracy:.4f}")
    
#     # Convert and save model
#     print("Converting model to TFLite...")
#     convert_to_tflite(model)
#     print("Model converted and saved successfully!")

# if __name__ == "__main__":
#     main() 

import os
import numpy as np
import tensorflow as tf
import kagglehub
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import pandas as pd

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

def load_and_preprocess_data():
    # Download the dataset
    path = kagglehub.dataset_download("datamunge/sign-language-mnist")
    
    # Load training data
    train_data = pd.read_csv(os.path.join(path, "sign_mnist_train.csv"))
    test_data = pd.read_csv(os.path.join(path, "sign_mnist_test.csv"))
    
    # Separate features and labels
    y_train = train_data['label']
    X_train = train_data.drop('label', axis=1)
    y_test = test_data['label']
    X_test = test_data.drop('label', axis=1)
    
    # Print unique label values to understand the issue
    print("Unique label values in training set:", np.unique(y_train))
    print("Unique label values in test set:", np.unique(y_test))
    
    # The dataset uses labels 0-9 for digits and 10-25 for letters A-Z except J and Z
    # Since we only handle 24 classes, we need to ensure we have the correct mapping
    # Adjust labels if needed or handle the specific classes correctly
    
    # For Sign Language MNIST, typically class labels are 0-23 (24 classes)
    # But the dataset might include J (9) and Z (25) which are usually excluded
    # Let's fix by ensuring all labels are in range 0-23
    
    # Filter out classes that we're not including (use this if needed)
    # Here I'm demonstrating filtering out any labels ≥ 24 if they exist
    train_mask = y_train < 24
    X_train = X_train.loc[train_mask].values
    y_train = y_train.loc[train_mask].values
    
    test_mask = y_test < 24
    X_test = X_test.loc[test_mask].values
    y_test = y_test.loc[test_mask].values
    
    # Reshape and normalize the data
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=24)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=24)
    
    return X_train, y_train, X_test, y_test

def create_model():
    model = models.Sequential([
        # First Convolutional Layer
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Second Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Third Convolutional Layer
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(24, activation='softmax')
    ])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    # Compile the model
    model.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    return history

def convert_to_tflite(model, X_train):
    # Convert to TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    # Representative dataset generator
    def representative_dataset_gen():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1]]
    
    converter.representative_dataset = representative_dataset_gen
    
    # Convert the model
    tflite_model = converter.convert()
    
    # Save the model
    with open('../src/model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Generate C array file for embedding in firmware
    with open('../src/model_data.c', 'w') as f:
        f.write('const unsigned char model_tflite[] = {\n')
        for i, byte in enumerate(tflite_model):
            f.write(f'0x{byte:02x}, ')
            if (i + 1) % 12 == 0:
                f.write('\n')
        f.write('\n};\n')
        f.write(f'const int model_tflite_len = {len(tflite_model)};\n')

def main():
    # Create necessary directories
    os.makedirs('../src', exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test = load_and_preprocess_data()
    
    # Create and train model
    print("Creating and training model...")
    model = create_model()
    history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Convert and save model
    print("Converting model to TFLite...")
    convert_to_tflite(model, X_train)
    print("Model converted and saved successfully!")

if __name__ == "__main__":
    main()