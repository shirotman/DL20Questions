# Imports
import torch
from peft import LoraConfig, get_peft_model
from transformers import pipeline, set_seed
from transformers import GPT2Tokenizer, GPT2Model
from transformers import GPT2Config, GPT2LMHeadModel
import os

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
model.load_state_dict(torch.load("./GPT2_ft2prompt.pth"))

# Generating a question
prompt = input("Insert a guess to prompt a question:")
generator = pipeline('text-generation', model = model, tokenizer=tokenizer)
print(generator("Prediction = " + prompt + ". Is it", max_new_tokens=5, num_return_sequences=5))

# Deterministic Inference time

most_probable_word= ""
question= ""
i=0
while most_probable_word != "?":

    i+=1

    det_prompt = "Prediction = " + prompt + ". Is it" + most_probable_word

    det_prompt_encode = tokenizer.encode(det_prompt, return_tensors='pt')

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(det_prompt_encode)

    # Get the predictions
    predictions = outputs.logits

    # Get the most probable word
    _, top_idx = torch.topk(predictions[0, -1, :], 1)

    # Decode the index to get the word
    most_probable_word = tokenizer.decode([top_idx.item()])

    # Get the probability of the most probable word
    prob = torch.nn.functional.softmax(predictions[0, -1, :], dim=-1)
    most_probable_word_prob = prob[top_idx].item()
    perplexity = torch.exp(-torch.sum(prob * torch.log(prob)))

    print(f"The most probable word is '{most_probable_word}' with a probability of {most_probable_word_prob}, The preplexity is equal to {perplexity}")
    question = question + most_probable_word

    if i>10:
        break

print("The final text is:")
print("Prediction = " + prompt + ". Is it" + question)