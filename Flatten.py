import torch
from transformers import AutoModel

def flatten_parameters_by_encoder_layer(model, num_encoder_layers=12):
    flattened_params_by_layer = {}
    for name, param in model.named_parameters():
        if "roberta.encoder.layer" in name:
            layer_name = ".".join(name.split(".")[:4])  # Adjust the number based on your specific layer depth
            if layer_name not in flattened_params_by_layer:
                flattened_params_by_layer[layer_name] = []
            flattened_params_by_layer[layer_name].append(param.view(-1))
    
    # Concatenate the flattened parameters for each encoder layer
    for layer_name, params in flattened_params_by_layer.items():
        flattened_params_by_layer[layer_name] = torch.cat(params)
    
    return flattened_params_by_layer

##make model from each checkpoint

val_set_size = 7654
models =[]

checkpoint_path = ""

for i in range(1,33):
    current_checkpoint = i * val_set_size 
    models[i] = AutoModel.from_pretrained(checkpoint_path + "checkpoint-" + string(epochs) + current_checkpoint)

flattened_layers = []

for model in models:
    flatten_parameters_by_encoder_layer(m)

###per layer and per model build matrix
##TODO horizontally?