from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM 
from transformers import DataCollatorForLanguageModeling  
from transformers import TrainingArguments, Trainer
from typing import Dict, Any

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForMaskedLM.from_pretrained("roberta-base")

dataset = load_dataset("cc100", lang="yo")

SEED = 42
dataset = dataset.shuffle(SEED)
#select only the first 5% of the dataset

temp = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=SEED)
dataset["train"] = temp["train"]
dataset["test"] = temp["test"]

temp = dataset["test"].train_test_split(test_size=0.5, shuffle=True, seed=SEED)
dataset["test"] = temp["train"]
dataset["val"] = temp["test"]

MLM_PROBABILITY = 0.1

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=MLM_PROBABILITY)

MAX_SEQ_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

# TODO: define a preprocessing function to tokenize a sample
def preprocess_function(sample: Dict[str, Any], seq_len: int):
    """
    Function applied to all the examples in the Dataset (individually or in batches). 
    It accepts as input a sample as a dictionary and return a new dictionary with the BERT tokens for that sample

    Args:
        sample Dict[str, Any]:
            Dictionary of sample
            
    Returns:
        Dict: Dictionary of tokenized sample in the following style:
        {
          "input_ids": list[int] # token ids
          "attention_mask": list[int] # Mask for self-attention (padding tokens are ignored).
        }
        Hint: if your are using the Huggingface tokenizer implementation, this is the default output format but check it yourself to be sure!
    """
    # set pad to eos
    # tokenizer.pad_token = tokenizer.eos_token
    Dict = tokenizer(sample["text"], truncation=True, padding="max_length", max_length=seq_len)
    return Dict


encoded_ds = dataset.map(
    preprocess_function, batched=True, fn_kwargs={"seq_len": MAX_SEQ_LENGTH}
)

print(len(encoded_ds))

split_names = encoded_ds.keys()
print("Split names:", split_names)
for k in split_names:
    print(len(encoded_ds[k]))

trainingArgs = TrainingArguments(
 evaluation_strategy="epoch",
 output_dir="./results_scratch_nohup_gpu4",
 num_train_epochs = 32,
 save_strategy="epoch",
 seed = SEED,
)

trainer = Trainer(
    model=model,
    args=trainingArgs,
    train_dataset=encoded_ds["train"],
    eval_dataset=encoded_ds["val"],
    data_collator=collator,
)

checkpoint = "./results_scratch_nohup_gpu2/checkpoint-214312"

if checkpoint:

    trainer.train(resume_from_checkpoint=checkpoint)
else:
    trainer.train()



trainer.evaluate()

