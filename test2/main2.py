from base64 import encode
import json
import random
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer
model = AutoModelForCausalLM.from_pretrained(
    "ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf",
    trust_remote_code=True, torch_dtype="auto", device_map="cuda", low_cpu_mem_usage=True

)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id

#dataset = load_dataset("timdettmers/openassistant-guanaco")
new_elements = list()
with open('mapping.json', 'r') as f:
    data = json.load(f)
    for element in data:
        print(element)
        new_ele = {
            "input": "### Instruction:\nSortiere die Elemente zu den Kategorien z.B. 2=c.\n### Input:\n" + element["input"]+"\n",
            "output": "### Response:\n" + element["output"]
        }
        #new_ele = "<s>### Instruction:\nUse the provided input to create an json.\n### Input:\n" + element["instruction"] + "\n### Response:\n" + element["output"] +"</s>"
        new_elements.append(new_ele)
        print(new_ele)
        pass

dataset = Dataset.from_list(new_elements)

# Use the loaded data here
# Example: print(data)

#from pandas import read_excel
#ground_truth_df = read_excel(ground_truth, sheet_name = sheet)
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#tokenizer = AutoTokenizer.from_pretrained("BlackSamorez/Mixtral-8x7b-AQLM-2Bit-1x16-hf")
tokenizer = AutoTokenizer.from_pretrained("ISTA-DASLab/Mixtral-8x7B-Instruct-v0_1-AQLM-2Bit-1x16-hf")
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

#features_column = ground_truth_df['output_ground_truth']
#new_features_column = features_column.apply(preprocess_features)
#ground_truth_df['output_ground_truth'] = new_features_column

#df_train = dataset.sample(frac = 0.8)
#df_test = dataset.drop(df_train.index)
#print("Amount data for training: " + str(len(df_train)))
#print("Amount data for testing: " + str(len(df_test)))
#from datasets import Dataset, DatasetDict
#dataset = DatasetDict({
#    "train": Dataset.from_pandas(df_train),
#    "test": Dataset.from_pandas(df_test)
#    })

from datasets import concatenate_datasets
import numpy as np
"""tokenized_inputs = concatenate_datasets([dataset]).map(lambda x: tokenizer(x["input"], truncation=True), batched=True, remove_columns=["input", "output"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")
tokenized_targets = concatenate_datasets([dataset]).map(lambda x: tokenizer(x["output"], truncation=True), batched=True, remove_columns=["input", "output"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")"""

tokenized_inputs = concatenate_datasets([dataset]).map(lambda x: tokenizer(x["input"], truncation=True), batched=True, remove_columns=["input", "output"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
max_source_length = int(np.percentile(input_lenghts, 85))
print(f"Max source length: {max_source_length}")
tokenized_targets = concatenate_datasets([dataset]).map(lambda x: tokenizer(x["output"], truncation=True), batched=True, remove_columns=["input", "output"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
max_target_length = int(np.percentile(target_lenghts, 90))
print(f"Max target length: {max_target_length}")

cutoff_len=max_source_length+max_target_length+5

def encode(text, add_bos_token):
    result = tokenizer.encode(text, truncation=True, max_length=cutoff_len)
    # Check if the first two tokens are BOS
    if len(result) >= 2 and result[:2] == [tokenizer.bos_token_id, tokenizer.bos_token_id]:
        result = result[1:]

    if not add_bos_token and result[0] == tokenizer.bos_token_id:
        result = result[1:]
    return result

def tokenize(prompt, append_eos_token=False):

    input_ids = encode(prompt, True)

    if append_eos_token and input_ids[-1] != tokenizer.eos_token_id and len(input_ids) < cutoff_len:
        input_ids.append(tokenizer.eos_token_id)

    input_ids = [tokenizer.pad_token_id] * (cutoff_len - len(input_ids)) + input_ids
    labels = [1] * len(input_ids)

    input_ids = torch.tensor(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": input_ids.ne(tokenizer.pad_token_id),
    }

def generate_and_tokenize_prompt(data_point):
    prompt = "<s>" + data_point["input"]+data_point["output"]
    return tokenize(prompt, True)

#tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset = dataset.map(generate_and_tokenize_prompt, new_fingerprint='%030x' % random.randrange(16**30))
tokenized_dataset.save_to_disk("data/train")

training_arguments = TrainingArguments(
        output_dir="./mixtral8x7b_aqlm_lora",
        #evaluation_strategy="steps",
        do_eval=False,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        #per_device_eval_batch_size=4,
        log_level="debug",
        logging_steps=25,
        learning_rate=0.003,
        eval_steps=25,
        save_strategy='steps',
        max_steps=15,
        #warmup_steps=25,
        lr_scheduler_type="constant",
)


peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate", "w1", "w2", "w3"]
)

trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset,
        #eval_dataset=dataset['test'],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=256,
        tokenizer=tokenizer,
        args=training_arguments,
)

trainer.train()

peft_model_id="results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)