import copy
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "results"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()

print("Peft model loaded")

sample = "### Instruction:\nSortiere die Elemente zu den Kategorien z.B. 2=c.\n### Input:Kategorien:\n10=Tiere\n20=Pflanzen\n30=Automarken\n40=CPUs\n\nSortiere:\nad=Gold\nbd=Metall\ncd=Papier\ndd=Erde\ned=Daimler.\n"

input_ids = tokenizer(sample, return_tensors="pt", truncation=True).input_ids.cuda()
# with torch.inference_mode():
outputs = model.generate(input_ids=input_ids, max_new_tokens=512, do_sample=True, top_p=0.9)
print(f"input sentence: {sample}\n{'---'* 20}")

print(f"summary:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")