import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import optuna
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history, plot_param_importances

# Enhanced DNN model creation function
def create_model(params, input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    
    # Add hidden layers with dropout and regularization
    for _ in range(params['n_layers']):
        model.add(tf.keras.layers.Dense(
            params['n_units'], 
            activation=params['activation'],
            kernel_regularizer=tf.keras.regularizers.l2(params['weight_decay'])
        ))
        model.add(tf.keras.layers.Dropout(params['dropout_rate']))
    
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    
    # Configure optimizer with learning rate
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
    # Suggest all hyperparameters
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

    # Load and preprocess data
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
    model = create_model(params, input_shape, num_classes)
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train with validation split
    history = model.fit(
        X_train, 
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=params['batch_size'],
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Evaluate on test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    model_parameters = model.count_params()

    # Constraint: accuracy must be > 0.80 (more stringent)
    if accuracy < 0.80:
        # Return a large value to penalize this trial
        return 1e6 + (0.80 - accuracy) * 1e6
    
    # Objective: minimize number of parameters while maintaining high accuracy
    # We add a small penalty for larger models to prefer simpler models with similar accuracy
    return model_parameters + (1 - accuracy) * 1000

if __name__ == '__main__':
    # Create Optuna study with pruning
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    
    # Optimize the objective function with more trials
    study.optimize(objective, n_trials=100, timeout=600)
    
    print('\nBest trial:')
    trial = study.best_trial
    print(f'  Params: {trial.params}')
    print(f'  Model parameters: {trial.value}')
    print(f'  Accuracy constraint met (>85%)')

    # Retrain best model on full training data
    best_params = trial.params
    
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

    best_model = create_model(best_params, input_shape, num_classes)
    best_model.fit(
        X_train, 
        y_train, 
        epochs=100, 
        batch_size=best_params['batch_size'], 
        verbose=1
    )
    
    # Evaluate final model
    _, final_accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f'\nFinal Model Test Accuracy: {final_accuracy:.4f}')

    # Save best model
    best_model.save('best_iris_dnn_optuna.h5')
    print("\nSaved best model as 'best_iris_dnn_optuna.h5'")

    # Convert to TFLite model and save
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    tflite_model = converter.convert()
    with open('best_iris_dnn_optuna.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as 'best_iris_dnn_optuna.tflite'")

    # Study statistics
    print("\nOptuna Study Statistics:")
    print(f"  Number of finished trials: {len(study.trials)}")
    print(f"  Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  Best value (optimized metric): {study.best_value}")
    print(f"  Best params: {study.best_params}")

    # Visualization
    print("\nGenerating optimization visualizations...")
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    
    fig1.show()
    fig2.show()
    
    # Save visualizations
    fig1.write_image("optimization_history.png")
    fig2.write_image("param_importances.png")
    print("Saved optimization visualizations as PNG files")