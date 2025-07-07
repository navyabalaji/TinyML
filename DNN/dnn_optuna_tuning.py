import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import optuna
import matplotlib.pyplot as plt

# DNN model creation function (same as before)
def create_model(n_hidden_layers, neurons_per_layer, input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for _ in range(n_hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def objective(trial):
    # Suggest hyperparameters
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 3)
    neurons_per_layer = trial.suggest_categorical('neurons_per_layer', [8, 16, 32, 64])

    # Load and preprocess data (same as before)
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y))

    # Build and train model
    model = create_model(n_hidden_layers, neurons_per_layer, input_shape, num_classes)
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    model_parameters = model.count_params()

    # Constraint: accuracy must be > 0.80
    if accuracy < 0.80:
        # Return a large value to penalize this trial
        return 1e6 + (0.80 - accuracy) * 1e6
    # Objective: minimize number of parameters
    return model_parameters

if __name__ == '__main__':
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    # Optimize the objective function
    study.optimize(objective, n_trials=30)

    print('Best trial:')
    trial = study.best_trial
    print(f'  Params: {trial.params}')
    print(f'  Model parameters: {trial.value}')
    print(f'  Accuracy constraint met (>80%)')

    # --- Additional Features ---
    # 1. Retrain best model on the same data
    best_n_hidden_layers = trial.params['n_hidden_layers']
    best_neurons_per_layer = trial.params['neurons_per_layer']

    # Reload and preprocess data
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_shape = (X_train.shape[1],)
    num_classes = len(np.unique(y))

    best_model = create_model(best_n_hidden_layers, best_neurons_per_layer, input_shape, num_classes)
    best_model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

    # 2. Convert to TFLite model and save
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()
    with open('best_iris_dnn_optuna.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'best_iris_dnn_optuna.tflite'")

    # 3. Print Optuna study statistics
    print("\nOptuna Study Statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Best value (min model params): {study.best_value}")
    print(f"  Best params: {study.best_params}")

    # 4. Plot graph of trial number vs. objective value
    values = [t.value for t in study.trials if t.value is not None]
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(values)+1), values, marker='o')
    plt.xlabel('Trial Number')
    plt.ylabel('Model Parameters (Objective Value)')
    plt.title('Optuna Optimization: Model Parameters per Trial')
    plt.grid(True)
    plt.tight_layout()
    plt.show() 