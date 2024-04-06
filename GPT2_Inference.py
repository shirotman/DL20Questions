# Imports
import torch
from peft import LoraConfig, get_peft_model
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model
from transformers import GPT2Config, GPT2LMHeadModel

# Loading model, tokenizer and configuration
configuration = GPT2Config()
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load("./GPT2_ft2prompt_friends.pth"))

# Generating a question
prompt = input("Insert a guess to prompt a question:")
generator = pipeline('text-generation', model = model, tokenizer=tokenizer)
print(generator("Prediction = " + prompt + ". Is it", max_new_tokens=5, num_return_sequences=5))