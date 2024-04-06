from transformers import pipeline, AutoModelForSeq2SeqLM, T5Tokenizer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
from sentence_transformers import SentenceTransformer, util
from peft import LoraConfig, get_peft_model
import torch
import random
import re
import numpy as np
import warnings
warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Choose an existing LLM 
# questions_model_name = "google/flan-t5-base"
questions_model_name = "distilgpt2"
# questions_model_name = "./peft-object-identification-training-1712241338/"
lora_config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

# Use a pre-trained version instead
objects_model_name = "./peft-object-identification-training"

# Load the model and tokenizer
questions_tokenizer = GPT2Tokenizer.from_pretrained(questions_model_name)
questions_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

questions_model = GPT2LMHeadModel.from_pretrained(questions_model_name)
questions_model = get_peft_model(questions_model, lora_config)
questions_model.resize_token_embeddings(len(questions_tokenizer))
questions_model.load_state_dict(torch.load("./GPT2_ft2prompt_friends.pth", map_location=torch.device(device)))
questions_model.config.use_cache = False

objects_model = AutoModelForSeq2SeqLM.from_pretrained(objects_model_name).to(device)
objects_tokenizer = T5Tokenizer.from_pretrained(objects_model_name, device=device)


# Load a pre-trained SBERT model to act as a sentence similarity model
similarity_model = SentenceTransformer("paraphrase-MiniLM-L6-v2").to(device)

questions_generator = pipeline('text-generation', model=questions_model, tokenizer=questions_tokenizer)
# generate question based on prompt
def generate_question(curr_guess=None):
  prompt_prefix = "Prediction = unknown. "
  if curr_guess:
    print("Prediction = " + curr_guess+ ". ")
    prompt_prefix = "Prediction = " + curr_guess+ ". "
  question_prefix = np.random.choice(["Is it", "Does it", "Can it", "Would you", "Are you likely to"], 1, p=[0.4,0.25,0.25,0.05,0.05])[0]
  prompt = prompt_prefix + question_prefix
#   print(prompt)
  generator_txt = questions_generator(prompt, max_new_tokens=8, num_return_sequences=4)
  generated_questions = [q['generated_text'][len(prompt_prefix):].split('?')[0]+'?' for q in generator_txt]
#   print()
#   for q in generator_txt:
#     generated_questions.append(q['generated_text'][len(prompt_prefix):].split('?')[0]+'?')
  return generated_questions

# update concept based on answer
def update_concept(concept, question, answer):
  print("Curr: "+concept)
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
  print("New: "+new_concept)
  return new_concept
# f"There is an object which is something in the real world and {concept[len('UNKNOWN and '):].lstrip()}. What would you guess this object to be?"
# guess the object based on user's answer(s)
def generate_guess(concept):
  start_prompt = 'Answer the question below:\nI am playing a guessing game where the answer is an object from the real world. If '
  end_prompt = 'then what is probably the object? '
  prompt = start_prompt + concept + end_prompt
  print(prompt)
  # start_prompt = "Answer the following question.\nIf the answer of "
  # end_prompt = f", then what is the real world object you guess it would probably relate to if it is {concept[len('UNKNOWN and '):].lstrip()}?\n\nThe object: "
  # prompt = start_prompt + f'\'{question}\'' + ' is ' + f'\'{answer}\'' + end_prompt
  
  input_ids = objects_tokenizer.encode(prompt, return_tensors="pt").to(device)
  output = objects_model.generate(input_ids, num_beams=2, max_new_tokens=16).to(device) 
  return objects_tokenizer.decode(output[0], skip_special_tokens=True)

def choose_new_question(existing_questions, new_options, similarity_model):
  # print(new_options)
  if len(existing_questions) == 0:
    return new_options[0]
  # Encode questions into embeddings
  embeddings_existing = similarity_model.encode(existing_questions, convert_to_tensor=True).to(device)
  embeddings_new = similarity_model.encode(new_options, convert_to_tensor=True).to(device)

  # Calculate cosine similarity scores
  similarity_scores = util.pytorch_cos_sim(embeddings_new, embeddings_existing)

  # Compute the average similarity for each question in new_options
  average_similarity_scores = similarity_scores.mean(dim=1)

  # Find the index of the least similar question in new_options
  least_similar_index = average_similarity_scores.argmin().item()
  least_similar_question = new_options[least_similar_index]
  return least_similar_question

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
    # generated_questions = generate_question(f"Think of a yes-no question (like 'is it...?', 'does it...?', 'can it...?', etc.) that would be most helpful in guessing a target object which is {concept}")
    # generated_questions = generate_question(f"Think of an object from the real world and formulate a relevant yes-no question that can help guessing this object, for example 'Is it a kind of a vegetable?', 'Can you play with it?' or 'Does it bark?'.")
    # print(generated_questions)
  else:
    generated_questions = generate_question(curr_guess)
    generated_questions = [q.replace(curr_guess, "it", 1).replace("the same as", "similar to").replace(" a it", " it").replace(" an it", " it") for q in generated_questions if not re.search(f"^Is it (a|an)?\s*{curr_guess}\?", q)] 
#   question = choose_new_question(questions, generated_questions, similarity_model)
  question = generated_questions[1]
#   print(question)
  questions.append(question)

  print(f"LLM asks: {question}")
  print("Your answer (yes/no): ")
  answer = input()

  if re.search(f"^Is it (a|an)?\s*{curr_guess}\?", question) and answer=="yes":  # The LLM guessed correctly in less than 20 rounds
    print(f"LLM guessed the object: {curr_guess} correctly in {i+1} rounds!")
    #Do some updates to the model
    break

  # Update concept based on answer
  concept = update_concept(concept, question, answer)
  curr_guess = generate_guess(concept)
  if i == question_limit - 1:
    print(f"LLM guessed the object: {curr_guess}")
    answer = input("Is this the object you have thought about?")

