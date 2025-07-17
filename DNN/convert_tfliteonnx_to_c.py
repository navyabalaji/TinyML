import os
import numpy as np

def flatten_and_format(name, array):
    flat = array.flatten()
    c_array = ', '.join([f"{x:.6f}" for x in flat])
    return f"const float {name}[{len(flat)}] = {{\n    {c_array}\n}};\n"

def generate_layer_structs(var_names, layer_shapes):
    struct_lines = []
    for i, (W_name, B_name, (in_dim, out_dim)) in enumerate(zip(var_names['weights'], var_names['biases'], layer_shapes)):
        struct_lines.append(f'{{ .type = LAYER_DENSE, .dense = {{ {in_dim}, {out_dim}, {W_name}, {B_name} }} }}')
        struct_lines.append('{ .type = LAYER_RELU }')
    struct_lines.pop()  # remove last RELU
    struct_lines.append('{ .type = LAYER_SOFTMAX }')
    return "const Layer model_layers[] = {\n    " + ",\n    ".join(struct_lines) + "\n};\n"

def write_c_file(output_c_path, arrays, layer_struct, num_layers):
    final_c = (
        '#include "ml_model.h"\n'
        '#include "model_data.h"\n\n' +
        arrays +
        "\n" +
        layer_struct +
        f"""
const NeuralNet model = {{
    .num_layers = {num_layers},
    .layers = model_layers
}};
"""
    )
    with open(output_c_path, "w") as f:
        f.write(final_c)
    print(f"[✔] Generated: {output_c_path}")

def process_tflite_model(path):
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()

    arrays = ""
    var_names = {'weights': [], 'biases': []}
    layer_shapes = []
    i = 0
    while i < len(details):
        name_w, data_w = details[i]['name'], interpreter.get_tensor(details[i]['index'])
        name_b, data_b = details[i+1]['name'], interpreter.get_tensor(details[i+1]['index'])
        W_var = f"W{i//2+1}"
        B_var = f"B{i//2+1}"
        arrays += flatten_and_format(W_var, data_w) + "\n"
        arrays += flatten_and_format(B_var, data_b) + "\n"
        var_names['weights'].append(W_var)
        var_names['biases'].append(B_var)
        layer_shapes.append((data_w.shape[0], data_w.shape[1]))
        i += 2
    struct_block = generate_layer_structs(var_names, layer_shapes)
    write_c_file("generated_model_data.c", arrays, struct_block, len(struct_block.split('{ .type = ')[1:]))

def process_onnx_model(path):
    import onnx
    import onnx.numpy_helper

    model = onnx.load(path)
    arrays = ""
    var_names = {'weights': [], 'biases': []}
    layer_shapes = []

    idx = 1
    for initializer in model.graph.initializer:
        data = onnx.numpy_helper.to_array(initializer)
        if data.ndim == 2:
            name = f"W{idx}"
            arrays += flatten_and_format(name, data) + "\n"
            var_names['weights'].append(name)
            shape = (data.shape[0], data.shape[1])
        elif data.ndim == 1:
            name = f"B{idx}"
            arrays += flatten_and_format(name, data) + "\n"
            var_names['biases'].append(name)
            layer_shapes.append(shape)
            idx += 1

    struct_block = generate_layer_structs(var_names, layer_shapes)
    write_c_file("generated_model_data.c", arrays, struct_block, len(struct_block.split('{ .type = ')[1:]))

def convert_model_to_c(input_model_path):
    ext = os.path.splitext(input_model_path)[1].lower()
    if ext == ".tflite":
        process_tflite_model(input_model_path)
    elif ext == ".onnx":
        process_onnx_model(input_model_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .tflite or .onnx file.")

if __name__ == "__main__":
    # Change filename below if needed
    convert_model_to_c("best_iris_dnn_optuna.onnx")
