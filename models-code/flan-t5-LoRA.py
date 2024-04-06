from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np
import evaluate
import os
import sys
sys.path.insert(0, '..')
from utils.preprocess_dataset import get_preprocessed_20Q_dataset

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset and LLM
dataset = get_preprocessed_20Q_dataset()
model_name='google/flan-t5-large'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def tokenize_function(example):
    start_prompt = 'Answer the question below:\nI am playing a guessing game where the answer is an object from the real world. If '
    end_prompt = ', then what is probably the object? '
    prompt = [start_prompt + f'{q.rstrip("?").lower().split()[1]}' + ' ' + (q.rstrip("?").lower().split()[0] if a=="no" else '') + (' not ' if a=="no" else '') + f'{" ".join(q.rstrip("?").lower().split()[2:])}' + end_prompt for q,a in zip(example["question"],example["answer"])]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["subject"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['subject', 'question', 'answer',])

# Perform Parameter Efficient Fine-Tuning (PEFT)
lora_config = LoraConfig(
    r=64, # Rank
    lora_alpha=128,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    use_rslora=True,
    # bias="lora_only",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config).to(device)

# Train PEFT Adapter
output_dir = os.path.join("..","fine-tuned-models","peft-object-identification-training")

peft_training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    predict_with_generate=True,
    learning_rate=1e-4,
    num_train_epochs=30,
    logging_steps=1,
    max_steps=200,
)

label_pad_token_id = -100   # ignore tokenizer pad token in the loss
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=peft_model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)
    
peft_trainer = Seq2SeqTrainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator
)

peft_trainer.train()

peft_model_path=output_dir
peft_model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)
# Uncomment following couple of lines to save a local instance of the complete model
# fpath = f"{output_dir}/flan-t5-LoRA.pth"
# torch.save(peft_trainer.model.state_dict(), fpath)



# Uncomment following lines to print some samples for human evaluation

# peft_t_model = peft_trainer.model.to(device)
# peft_model = peft_model.to(device)
# original_model = original_model.to(device)


# for index in np.random.choice(range(len(dataset['test'])), 5):
#     i = index.item()
#     question = dataset['test'][i]['question']
#     answer = dataset['test'][i]['answer']
#     object = dataset['test'][i]['subject']
#     sentenced_question = question.rstrip("?").lower().split()
#     sentenced_question = sentenced_question[1] + ' ' + (sentenced_question[0] if answer=="no" else '') + (' not ' if answer=="no" else '') + " ".join(sentenced_question[2:])

#     prompt = f"""Answer the question below:
#     I am playing a guessing game where the answer is an object from the real world.
#     If {sentenced_question}, then what is probably the object? """

#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

#     original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=15, num_beams=1)).to(device)
#     original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

#     peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=15, num_beams=1))
#     peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)
#     peft_t_model_outputs = peft_t_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=15, num_beams=1))
#     peft_t_model_text_output = tokenizer.decode(peft_t_model_outputs[0], skip_special_tokens=True)
#     dash_line = '-'.join('' for x in range(100))
#     print(prompt)
#     print(dash_line)
#     print(f'GIVEN OBJECT:\n{object}')
#     print(dash_line)
#     print(f'ORIGINAL MODEL:\n{original_model_text_output}')
#     print(dash_line)
#     print(f'PEFT T MODEL: {peft_t_model_text_output}')
#     print(f'PEFT MODEL: {peft_model_text_output}')
#     print()


torch.cuda.empty_cache()
