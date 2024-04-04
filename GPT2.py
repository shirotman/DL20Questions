import os
#os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch.nn as nn
#import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


# Load GPT2

from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
print(generator("Is it a ,", max_length=10, num_return_sequences=5))

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
model = GPT2Model.from_pretrained('distilgpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

# Fine Tune

from transformers import GPT2Config, GPT2Model

# Initializing a GPT2 configuration
configuration = GPT2Config()

# Initializing a model (with random weights) from the configuration
#model = GPT2Model(configuration)

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Model #, DistilGPT2Model
#import accelerate
model = GPT2LMHeadModel.from_pretrained("distilgpt2")#, load_in_8bit=True, device_map='auto')

def print_trainable_parameters(model): # Print number of trainable parameters
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# LORA
from datasets import load_dataset

data = load_dataset("clips/20Q")
#data_tensor= data
#data_tensor = data.map(lambda examples: {'subject': torch.tensor(examples['subject']), 'question': torch.tensor(examples['question']), 'label': torch.tensor(examples['label']), 'label_fine_grained': torch.tensor(examples['label_fine_grained'])})
# train_data= data["train"]
# test_data= data["test"]

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

# class CastOutputToFloat(nn.Sequential):
#     def forward(self, x): return super().forward(x).to(torch.float32)
# model.lm_head = CastOutputToFloat(model.lm_head)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)
for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later

model = get_peft_model(model, config)
print_trainable_parameters(model)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
#  Data Pre-process
def merge_columns(example):
    example["prediction"] = example["question"] + " ->: " + example["label_fine_grained"]
    #return tokenizer(example["prediction"], truncation=True, padding='max_length', max_length=512)
    return example

import transformers

data['train'] = data['train'].map(merge_columns)
#print(data['train']["prediction"][:5])
data['train'][0]
#
data_train = data['train'].map(lambda samples: tokenizer(samples['prediction'], truncation=True, padding='max_length', max_length=512), batched=True)
# data_train = data_train.remove_columns('label')
data_tensor = data_train.map(lambda examples: {'input_ids': torch.tensor(examples['input_ids']), 'attention_mask': torch.tensor(examples['attention_mask'])})
data_tensor = data_tensor.remove_columns(['label','subject','question','label_fine_grained','prediction'])

# model.base_model.model.lm_head[0].requires_grad = True
for param in model.base_model.model.lm_head.parameters():
    param.requires_grad = True
trainer = transformers.Trainer(
    model=model,
    train_dataset=data_tensor,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        #fp16=
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

fname = "./GPT2_fine_tuned.pth"
torch.save(model.state_dict(), fname)


print(6)