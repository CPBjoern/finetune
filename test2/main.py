import random
import sys
from datasets import load_dataset, Dataset
import json
import accelerate

#dataset = load_dataset("samsum")

ground_truth = "ground-truth.xlsx"
sheet = "main"
model_id="Mixtral-8x7B-Instruct-v0.1-GPTQ"
         
prompt_instruction="Input: "

new_elements = list()
"""with open('sandwich.json', 'r') as f:
    data = json.load(f)
    for element in data:
        print(element)
        new_ele = {
            "input": "### Instruction:\nUse the provided input to create an json.\n### Input:\n" + element["instruction"]+"\n",
            "output": "### Response:\n" + element["output"]
        }
        #new_ele = "<s>### Instruction:\nUse the provided input to create an json.\n### Input:\n" + element["instruction"] + "\n### Response:\n" + element["output"] +"</s>"
        new_elements.append(new_ele)
        print(new_ele)
        pass"""
    
"""with open('cp_infos.json', 'r') as f:
    data = json.load(f)
    for element in data:
        print(element)
        new_ele = {
            "input": "### Instruction:\nBenutze folgende Daten um Fragen zum Corporate Planner zu beantworten.\n### Input:\nDatentyp: " + element["datatype"]+"\nSektion: " + element["section"] + "\nInhalt: " + element["text"],
            "output": ""
        }
        #new_ele = "<s>### Instruction:\nUse the provided input to create an json.\n### Input:\n" + element["instruction"] + "\n### Response:\n" + element["output"] +"</s>"
        new_elements.append(new_ele)
        print(new_ele)
        pass"""

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
tokenizer = AutoTokenizer.from_pretrained(model_id)
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
#tokenized_dataset["test"].save_to_disk("data/eval")
print('''tokenized_dataset["train"][0]"''')
print(tokenizer.decode(tokenized_dataset[0]["input_ids"]))

from transformers import AutoModelForCausalLM

# huggingface hub model id
model_id="Mixtral-8x7B-Instruct-v0.1-GPTQ"
# load model from the hub
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, load_in_8bit=False, device_map="cpu")

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 #lora_alpha=32,
 lora_alpha=4,
 #target_modules=["q_proj", "o_proj", "k_proj", "v_proj"],
 target_modules=["k_proj", "v_proj"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.CAUSAL_LM
)
# prepare int-8 model for training
model = prepare_model_for_kbit_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
#data_collator = DataCollatorForSeq2Seq(
#    tokenizer,
#    model=model,
#    label_pad_token_id=label_pad_token_id,
#    pad_to_multiple_of=8
#)
from trl import SFTTrainer
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import json

output_dir="mylora"

# Define training args
"""training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
	auto_find_batch_size=True,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps",
    logging_steps=500,
    save_strategy="no",
    report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)
model.config.use_cache = False """

############################################################
trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
    args= TrainingArguments(
        #report_to="tensorboard",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1.0,
        #warmup_steps=math.ceil(warmup_steps / gradient_accumulation_steps),
        num_train_epochs=5,
        learning_rate=0.0003,
        #fp16=False if shared.args.cpu or shared.args.bf16 else True,
        bf16=True,
        output_dir=".",
        lr_scheduler_type="constant",
        logging_steps=10,
        log_level='info'
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

############################################################

model.config.use_cache = False

if torch.__version__ >= "2" and sys.platform != "win32":
    lora_model = torch.compile(model)

# train model
trainer.train()

peft_model_id="results"
trainer.model.save_pretrained(peft_model_id)
tokenizer.save_pretrained(peft_model_id)