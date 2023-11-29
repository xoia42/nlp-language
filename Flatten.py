import torch
from transformers import AutoModel
from numpy import linalg
import numpy as np

def flatten_parameters_by_encoder_layer(model, num_encoder_layers=12):
    flattened_params_by_layer = {}
    for name, param in model.named_parameters():
        if "encoder.layer" in name:
            layer_name = ".".join(name.split(".")[:3])  # Adjust the number based on your specific layer depth
            if layer_name not in flattened_params_by_layer:
                flattened_params_by_layer[layer_name] = []
            flattened_params_by_layer[layer_name].append(param.view(-1))
    
    # Concatenate the flattened parameters for each encoder layer
    for layer_name, params in flattened_params_by_layer.items():
        flattened_params_by_layer[layer_name] = torch.cat(params)
    
    return flattened_params_by_layer

##make model from each checkpoint

VAL_SET_SIZE = 7654
models =list()

checkpoint_path = "results_scratch_nohup_gpu/"

for i in range(1,33):
    current_checkpoint = i * VAL_SET_SIZE 
    path =checkpoint_path + "checkpoint-" + str(current_checkpoint)
    test = AutoModel.from_pretrained(path)
    f = flatten_parameters_by_encoder_layer(test)
    models.append(flatten_parameters_by_encoder_layer(AutoModel.from_pretrained(path)))

flattened_layers = []

#for i in range(len(models)):
#   models[i] = flatten_parameters_by_encoder_layer(models[i])

###per layer and per model build matrix
##TODO horizontally?
## TODO need to transpose v s?


matrices ={}

name = "encoder.layer"


for i in range(0,12):
    matrices[i] =[]
    for model in models:
        current_vector = model[name+f".{i}"]
        if len(matrices[i]) == 0:
            matrices[i] = current_vector.detach().numpy()
        else:
            matrices[i] = np.row_stack((matrices[i], current_vector.detach().numpy())) 
            #print(matrices[i].shape)  


print(len(matrices))
##svd per matrix

def get_v(matrix):
    u,s,vh = linalg.svd(matrix, full_matrices=False)
    print("Dimensions U:",u.shape)
    print("Dimensions S:",s.shape)
    print("Dimensions VH:",vh.shape)
    return vh

vs = {}
for key, matrix in matrices.items():
    vs[key] = get_v(matrix).T.conj()
    print("Dimensions V:",vs[key].shape)
##Transponieren



print(vs)