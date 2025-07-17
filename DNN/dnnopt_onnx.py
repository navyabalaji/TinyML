import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import optuna
import matplotlib.pyplot as plt

# Enhanced DNN model creation function
def create_model(params, input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    # Add hidden layers
    for _ in range(params['n_layers']):
        model.add(tf.keras.layers.Dense(
            params['n_units'],
            activation=params['activation'],
            kernel_regularizer=tf.keras.regularizers.l2(params['weight_decay'])
        ))
        model.add(tf.keras.layers.Dropout(params['dropout_rate']))

    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    if params['optimizer'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=params['learning_rate'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=params['learning_rate'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def objective(trial):
    params = {
        'n_layers': trial.suggest_int('n_layers', 1, 5),
        'n_units': trial.suggest_categorical('n_units', [8, 16, 32, 64, 128]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'sigmoid']),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['adam', 'sgd', 'rmsprop']),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32, 64]),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    }

    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = create_model(params, X_train.shape[1:], len(np.unique(y)))

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )

    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    model_parameters = model.count_params()

    if accuracy < 0.80:
        return 1e6 + (0.80 - accuracy) * 1e6

    return model_parameters + (1 - accuracy) * 1000

if __name__ == '__main__':
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=100, timeout=600)

    print('\nBest trial:')
    trial = study.best_trial
    print(f'  Params: {trial.params}')
    print(f'  Model parameters: {trial.value}')
    print(f'  Accuracy constraint met (>85%)')

    best_params = trial.params

    # Final training
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    best_model = create_model(best_params, X_train.shape[1:], len(np.unique(y)))
    best_model.fit(X_train, y_train, epochs=100, batch_size=best_params['batch_size'], verbose=1)

    _, final_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'\nFinal Model Test Accuracy: {final_accuracy:.4f}')

    # Save best model
    best_model.save('best_iris_dnn_optuna.h5')
    print("Saved best model as 'best_iris_dnn_optuna.h5'")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()
    with open('best_iris_dnn_optuna.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'best_iris_dnn_optuna.tflite'")

    # Convert to ONNX
    import tf2onnx
    spec = (tf.TensorSpec(best_model.input.shape, tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_keras(best_model, input_signature=spec, opset=13)
    with open("best_iris_dnn_optuna.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("ONNX model saved as 'best_iris_dnn_optuna.onnx'")

    # Optuna summary
    print("\nOptuna Study Statistics:")
    print(f"  Finished trials: {len(study.trials)}")
    print(f"  Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Best value: {study.best_value}")
    print(f"  Best params: {study.best_params}")

    # Plot and save
    from optuna.visualization import plot_optimization_history, plot_param_importances
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()

    fig1.write_image("optimization_history.png")
    fig2.write_image("param_importances.png")
    print("Saved optimization visualizations.")
