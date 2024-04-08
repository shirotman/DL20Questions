from transformers import pipeline, AutoModelForSeq2SeqLM, T5Tokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model
import torch
import random
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose an existing LLM 
questions_model_name = "distilgpt2"
lora_config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

# Use a pre-trained version instead
objects_model_name = "./fine-tuned-models/peft-object-identification-training"

# Load the model and tokenizer
questions_tokenizer = GPT2Tokenizer.from_pretrained(questions_model_name)
questions_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

questions_model = GPT2LMHeadModel.from_pretrained(questions_model_name)
questions_model = get_peft_model(questions_model, lora_config)
questions_model.resize_token_embeddings(len(questions_tokenizer))
questions_model.load_state_dict(torch.load("./fine-tuned-models/GPT2-LoRA/GPT2_ft2prompt.pth", map_location=torch.device(device)))
questions_model.config.use_cache = False

objects_model = AutoModelForSeq2SeqLM.from_pretrained(objects_model_name).to(device)
objects_tokenizer = T5Tokenizer.from_pretrained(objects_model_name, device=device)


# Load a pre-trained SBERT model to act as a sentence similarity model
similarity_model = SentenceTransformer("paraphrase-MiniLM-L6-v2").to(device)

questions_generator = pipeline('text-generation', model=questions_model, tokenizer=questions_tokenizer)
# generate question based on prompt
def generate_question(curr_guess=None):
  prompt_prefix = "Prediction = object. "
  if curr_guess:
    # print("Prediction = " + curr_guess + ". ")
    print()
    prompt_prefix = "Prediction = " + curr_guess+ ". "
  question_prefix = np.random.choice(["Is it", "Does it", "Can it", "Would you"], 1, p=[0.4,0.27,0.27,0.06])[0]
  prompt = prompt_prefix + question_prefix
  generator_txt = questions_generator(prompt, max_new_tokens=8, num_return_sequences=4)
  generated_questions = [q['generated_text'][len(prompt_prefix):].split('?')[0]+'?' for q in generator_txt]
  return generated_questions

# update concept based on answer
def update_concept(concept, question, answer):
  new_concept = concept.replace("UNKNOWN", "")
  words = question.rstrip("?").lower().split()  # Split the question into words
  if answer.lower() == "yes":
    swapped_words = words[1] + " " + words[0] + " " + " ".join(words[2:])
    new_concept = swapped_words + ", " + new_concept
  elif answer.lower() == "no":
    swapped_words = words[1] + " " + words[0] + " not " + " ".join(words[2:])
    new_concept = swapped_words + ", " + new_concept
  else:
    print("You've entered an illegal answer")
    return concept
  concept_parts = new_concept.split(',')
  if len(concept_parts) == 3:
    new_concept = f"{concept_parts[0]} and{concept_parts[1]}, "
  return new_concept

# guess the object based on user's answer(s)
def generate_guess(concept):
  start_prompt = 'Answer the question below:\nI am playing a guessing game where the answer is an object from the real world. If '
  end_prompt = 'then what is probably the object? '
  prompt = start_prompt + concept + end_prompt
  
  input_ids = objects_tokenizer.encode(prompt, return_tensors="pt").to(device)
  output = objects_model.generate(input_ids, num_beams=2, max_new_tokens=16).to(device) 
  return objects_tokenizer.decode(output[0], skip_special_tokens=True)


# Define the game loop
concept = "UNKNOWN"
curr_guess = None
question_limit = 20  # Maximum number of questions
questions = []

print("Please think of an object from the real world")
print("Once you have an object in mind, please enter anything to begin ")
input()
print()

for i in range(question_limit):
  generated_questions = None
  if curr_guess == None:
    generated_questions = generate_question()
  else:
    generated_questions = generate_question(curr_guess)
    generated_questions = [q.replace(curr_guess, "it", 1).replace("the same as", "similar to").replace(" a it", " it").replace(" an it", " it") for q in generated_questions if not re.search(f"^Is it (a|an)?\s*{curr_guess}\?", q)] 
  question = generated_questions[0]
  questions.append(question)

  print(f"LLM asks: {question}")
  print("Your answer (yes/no): ")
  answer = input()

  # Update concept based on answer
  concept = update_concept(concept, question, answer)
  curr_guess = generate_guess(concept)
  if i == question_limit - 1:
    print(f"LLM guessed the object: {curr_guess}")
    print("Is this the object you have thought about?")
    answer = input()

