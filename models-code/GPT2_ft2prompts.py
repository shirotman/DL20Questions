# Imports
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model
import sys
sys.path.insert(0, '..')
from utils.preprocess_dataset import get_preprocessed_20Q_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load GPT2 and Tokinizer
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

# Initializing a GPT2 configuration
configuration = GPT2Config()

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
model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

config = LoraConfig(
    r=16, #LoRA rank
    lora_alpha=32, #alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)
for param in model.parameters():
    param.requires_grad = False  # freeze the model

model = get_peft_model(model, config)

#  Data Pre-process
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

def merge_columns(example):
    example["prediction"] = "Prediction = " + example["subject"] + ". " + example["question"] + " ->: " + example["answer"]
    return example

data = get_preprocessed_20Q_dataset()
data['train'] = data['train'].map(merge_columns)
data_train = data['train'].map(lambda samples: tokenizer(samples['prediction'], truncation=True, padding='max_length', max_length=512), batched=True)
data_tensor = data_train.map(lambda examples: {'input_ids': torch.tensor(examples['input_ids']), 'attention_mask': torch.tensor(examples['attention_mask'])})
data_tensor = data_tensor.remove_columns(['subject','question','answer','prediction'])


# Training time
for param in model.base_model.model.lm_head.parameters():
    param.requires_grad = True
print_trainable_parameters(model)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data_tensor,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        max_steps=200,
        learning_rate=2e-4,
        logging_steps=1,
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()

# Saving the model
fname = "../fine-tuned-models/GPT2-LoRA/GPT2_ft2prompt.pth"
torch.save(model.state_dict(), fname)