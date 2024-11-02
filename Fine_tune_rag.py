from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import json
import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    """
    Custom Dataset for Question-Answer fine-tuning.
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item["context"]
        question = item["question"]
        answer = item["answer"]

        # Construct input text as "question + context"
        input_text = f"Question: {question} Context: {context}"
        target_text = answer

        # Tokenize input and target
        inputs = self.tokenizer(input_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")
        targets = self.tokenizer(target_text, max_length=self.max_length, truncation=True, padding="max_length", return_tensors="pt")

        # Return input_ids and labels for training
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": targets["input_ids"].squeeze(),
        }


def fine_tune_rag_model(data_path, model_path, output_dir):
    """
    Fine-tune the RAG model using the provided question-answer training data.
    """
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare the dataset
    dataset = QADataset(data_path, tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10,
        save_total_limit=2,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    # Path to the fine-tuning data and model
    data_path = "../data/fine_tune_data.json"
    model_path = "../models/fine_tuned_llama_model"
    output_dir = "../models/fine_tuned_llama_output"

    ## Fine-tune the model
    fine_tune_rag_model(data_path, model_path, output_dir)
    
