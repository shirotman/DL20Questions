from transformers import AutoModelForSeq2SeqLM, T5Tokenizer, GenerationConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch
import evaluate
from preprocess_dataset import get_preprocessed_20Q_dataset

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset and LLM
dataset = get_preprocessed_20Q_dataset()
model_name='google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

dash_line = '-'.join('' for x in range(100))


def tokenize_function(example):
    start_prompt = 'I am playing a guessing game where the answer is an object from the real world. If '
    end_prompt = ', then what is probably the object?'
    prompt = [start_prompt + f'{q.rstrip("?").lower().split()[1]}' + ' ' + (q.rstrip("?").lower().split()[0] if a=="no" else '') + (' not ' if a=="no" else '') + f'{" ".join(q.rstrip("?").lower().split()[2:])}' + end_prompt for q,a in zip(example["question"],example["answer"])]
    example['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["subject"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return example

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['subject', 'question', 'answer',])

# Perform Parameter Efficient Fine-Tuning (PEFT)
lora_config = LoraConfig(
    r=8, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    use_rslora=True,
    bias="lora_only",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config).to(device)

# Train PEFT Adapter
output_dir = f'./peft-object-identification-training'

peft_training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    predict_with_generate=True,
    learning_rate=1e-4,
    num_train_epochs=1,
    logging_steps=1,
    max_steps=100,
    load_best_model_at_end=True,
    evaluation_strategy="steps",
    save_strategy="steps" 
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
    eval_dataset=tokenized_datasets['test'],
    data_collator=data_collator
)

peft_trainer.train()

peft_model_path=output_dir

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)


torch.cuda.empty_cache()
