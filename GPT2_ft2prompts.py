# Imports
import torch
import transformers
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from datasets import concatenate_datasets, DatasetDict

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
from datasets import load_dataset

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)
for param in model.parameters():
    param.requires_grad = False  # freeze the model

model = get_peft_model(model, config)

#  Data Pre-process

user_input = input("clips dataset or friends?: ")

if user_input== "clips":
    data = load_dataset("clips/20Q")
elif user_input== "friends":
    data = load_dataset('csv', data_files="20q_friends.csv")

data = load_dataset("clips/20Q")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

def merge_columns(example):
    example["prediction"] = "Prediction = "+ example["subject"] + ". " + example["question"] + " ->: " + example["label_fine_grained"]
    return
def merge_columns_friends(example):
    example["prediction"] = "Prediction = "+ example["ground_truth"] + ". " + example["question"] + " ->: " + example["answer"]
    #return tokenizer(example["prediction"], truncation=True, padding='max_length', max_length=512)
    return example

def get_preprocessed_20Q_dataset():
    huggingface_dataset_name = "clips/20Q"
    dataset = load_dataset(huggingface_dataset_name)

    def map_labels(sample):
        label = sample["label"]
        sample["label"] = "yes" if label == True else "no"
        return sample

    combined_data = concatenate_datasets([dataset["train"], dataset["test"]])
    #combined_data = combined_data.remove_columns("label_fine_grained")
    combined_data = combined_data.map(map_labels)
    combined_data = combined_data.rename_column("label", "answer")

    train_test_dataset = combined_data.train_test_split(test_size=0.2)
    test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)

    train_test_valid_dataset = DatasetDict({
        'train': train_test_dataset['train'],
        'test': test_valid['test'],
        'validation': test_valid['train']}, random_state=42)

    return train_test_valid_dataset

data_train = get_preprocessed_20Q_dataset()['train']

if user_input== "clips":
    data_train = data_train.map(merge_columns)
    data_train = data_train.map(lambda samples: tokenizer(samples['prediction'], truncation=True, padding='max_length', max_length=512), batched=True)
    data_tensor = data_train.map(lambda examples: {'input_ids': torch.tensor(examples['input_ids']), 'attention_mask': torch.tensor(examples['attention_mask'])})
    data_tensor = data_tensor.remove_columns(['subject','question','label_fine_grained','prediction'])
elif user_input== "friends":
    data_train = data_train.map(merge_columns_friends)
    data_train = data_train.map(lambda samples: tokenizer(samples['prediction'], truncation=True, padding='max_length', max_length=512), batched=True)
    data_tensor = data_train.map(lambda examples: {'input_ids': torch.tensor(examples['input_ids']), 'attention_mask': torch.tensor(examples['attention_mask'])})
    data_tensor = data_tensor.remove_columns(['question_num','game_num','question','answer','prediction', 'ground_truth'])

    model.load_state_dict(torch.load("./GPT2_ft2prompt.pth"))


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

if user_input== "clips":
    fname = "./GPT2_ft2prompt.pth"
elif user_input== "friends":
    fname = "./GPT2_ft2prompt_friends.pth"

torch.save(model.state_dict(), fname)

