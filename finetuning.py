import json
import random
from transformers import AutoModel
from transformers import RobertaModel, RobertaTokenizer

path = '/home/fhradilak/nlp-language/results_scratch_nohup_gpu/checkpoint-244928'
test = AutoModel.from_pretrained(path)
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaModel.from_pretrained(model_name)



# Get the named parameters
params = []
for name, param in model.named_parameters():
    if 'encoder.layer.1' in name:  # Change index to access different layers
            params.append((".".join(name.split(".")[:3]), param.shape))

    #for i in range(0,11):
        # if f'encoder.layer.{i}' in name:  # Change index to access different layers
        #     print(f"Parameter name: {name}, Shape: {param.shape}")

# Freeze all parameters in the model
for param in model.parameters():
    param.requires_grad = False

unfreezed_params = {}

for layer in range(0,12):
    unfreezed_params[layer] = {}
    for p in range(0,16):
        layer_name = f'encoder.layer.{layer}.'
        #get random number between 0, params.length-1
        indx = random.randint(0, len(params)-1)
        random_param = params[indx]
        param_name = layer_name+random_param[0]
        if(param_name.contains("bias")):
            column = random.randint(0, random_param[1][0])
            unfreezed_params[layer][p] = (param_name, column)
            #TODO unfre
            #     for name, param in model.named_parameters():
            #       if name == param_to_unfreeze:
            #           param.requires_grad = True
            #           break
            model.roberta.param_name[column].requires_grad = True
        else:
            #TODO row column
            column = random.randint(0,random_param[1][1])
            row = random.randint(0,random_param[1][0])
            unfreezed_params[layer][p] = (param_name, (column,row))
            model.roberta.param_name[row][column].requires_grad = True
       
trainingArgs()
     
file_path = 'V.json'

with open(file_path, 'r') as json_file:
    loaded_vs = json.load(json_file)



