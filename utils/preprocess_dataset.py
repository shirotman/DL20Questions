from datasets import load_dataset, concatenate_datasets
import os

# This script/function is inteneded to merge data which was taken from 4 different sources:
# An existing HuggingFace dataset, two LLM (Gemini and GPT4) generated datasets and a dedicated dataset created by users

def get_preprocessed_20Q_dataset():
    local_datasets_path = os.path.join("..","local-datasets")
    huggingface_dataset_name = "clips/20Q"
    dataset = load_dataset(huggingface_dataset_name)

    def map_answers(sample):
        answer = sample["answer"]
        sample["answer"] = "yes" if (answer == True or answer == "win") else "no"
        return sample

    complete_remote_dataset = concatenate_datasets([dataset["train"], dataset["test"]])
    complete_remote_dataset = complete_remote_dataset.remove_columns("label_fine_grained")
    complete_remote_dataset = complete_remote_dataset.rename_column("label", "answer")
    complete_remote_dataset = complete_remote_dataset.map(map_answers)

    gemini_dataset = load_dataset("json", data_files=os.path.join(local_datasets_path,"gemini-dataset.jsonl"))["train"]
    gpt_dataset = load_dataset("json", data_files=os.path.join(local_datasets_path,"gpt4-dataset.jsonl"))["train"]

    friends_dataset = load_dataset("csv", data_files=os.path.join(local_datasets_path,"20q_friends.csv"))["train"]
    friends_dataset = friends_dataset.remove_columns(["question_num", "game_num"]).rename_column("ground_truth", "subject")
    friends_dataset = friends_dataset.map(map_answers)

    combined_data = concatenate_datasets([complete_remote_dataset, gemini_dataset, gpt_dataset, friends_dataset])

    train_test_datasets = combined_data.train_test_split(test_size=0.1, seed=42)

    return train_test_datasets