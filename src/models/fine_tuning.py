"""
Fine-tuning pipeline for academic summarization using LoRA.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import logging
from typing import Dict, Any, List, Optional
import os
import requests
import tempfile
import pdfplumber

class AcademicSummarizationFineTuner:
    """Fine-tuning pipeline for academic paper summarization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('base_model', 'facebook/bart-large-cnn')
        self.output_dir = config.get('output_dir', 'models/summarizer_lora')
        self.logger = logging.getLogger("fine_tuner")
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=config.get('lora_r', 16),
            lora_alpha=config.get('lora_alpha', 32),
            target_modules=config.get('target_modules', ["q_proj", "v_proj"]),
            lora_dropout=config.get('lora_dropout', 0.1),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = None
        self.tokenizer = None
        
    def prepare_model(self):
        """Load and prepare model for fine-tuning (no bitsandbytes/quantization for Mac)."""
        self.logger.info(f"Loading base model: {self.model_name}")

        # Load model and tokenizer (standard, no quantization)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        # Add LoRA adapters if needed
        if hasattr(self, 'lora_config') and self.lora_config is not None:
            self.model = get_peft_model(self.model, self.lora_config)

        # Print trainable parameters
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()

        self.logger.info("Model prepared for fine-tuning")

    def prepare_dataset(self, data_path: str) -> Dataset:
        """Prepare dataset for training."""
        self.logger.info(f"Loading dataset from: {data_path}")
        
        # Load data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Format for training
        formatted_data = []
        for item in data:
            # Create prompt-response pairs
            prompt = self.create_prompt(item['paper_text'], item.get('section_type', 'full'))
            response = item['summary']
            
            # Combine prompt and response
            full_text = f"{prompt}{response}{self.tokenizer.eos_token}"
            
            formatted_data.append({
                'text': full_text,
                'input_ids': None  # Will be tokenized later
            })
        
        # Tokenize
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=True,
                max_length=self.config.get('max_length', 1024),
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized['labels'] = tokenized['input_ids'].clone()
            
            return tokenized
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        self.logger.info(f"Dataset prepared with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def create_prompt(self, paper_text: str, section_type: str = 'full') -> str:
        """Create appropriate prompt for different types of summarization."""
        
        prompts = {
            'abstract': "Summarize the following abstract concisely:\n\n",
            'introduction': "Summarize the key points from this introduction:\n\n",
            'methodology': "Summarize the methodology described:\n\n",
            'results': "Summarize the main results and findings:\n\n",
            'conclusion': "Summarize the conclusions and implications:\n\n",
            'full': "Provide a comprehensive summary of this academic paper:\n\n"
        }
        
        prompt_prefix = prompts.get(section_type, prompts['full'])
        
        # Truncate paper text if too long
        max_input_length = self.config.get('max_input_length', 2000)
        if len(paper_text) > max_input_length:
            paper_text = paper_text[:max_input_length] + "..."
        
        return f"{prompt_prefix}{paper_text}\n\nSummary: "
    
    def create_sample_dataset(self, training_json_path: str, output_path: str):
        """Create a sample dataset by reading training_json_path, downloading PDF, extracting text, and replacing paper_text."""
        import requests
        import tempfile
        import pdfplumber
        import os
        import json

        # Read training data
        with open(training_json_path, 'r') as f:
            data = json.load(f)

        processed_data = []
        for entry in data:
            pdf_url = entry.get('pdf_url')
            summary = entry.get('summary')
            section_type = entry.get('section_type', 'abstract')
            paper_text = ""
            if pdf_url:
                try:
                    response = requests.get(pdf_url, timeout=30)
                    response.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                        tmp_pdf.write(response.content)
                        tmp_pdf_path = tmp_pdf.name
                    with pdfplumber.open(tmp_pdf_path) as pdf:
                        paper_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                    os.unlink(tmp_pdf_path)
                except Exception as e:
                    paper_text = f"[PDF extraction failed: {str(e)}]"
            else:
                paper_text = entry.get('paper_text', "")
            processed_data.append({
                "paper_text": paper_text.strip(),
                "summary": summary,
                "section_type": section_type,
                "pdf_url": pdf_url
            })

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        self.logger.info(f"Sample dataset created at {output_path}")
        return output_path
    
    def train(self, train_dataset: Dataset, eval_dataset: Optional[Dataset] = None):
        """Fine-tune the model."""
        self.logger.info("Starting fine-tuning")
        print("TrainingArguments class:", TrainingArguments)
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.config.get('num_epochs', 3),
            per_device_train_batch_size=self.config.get('batch_size', 4),
            per_device_eval_batch_size=self.config.get('eval_batch_size', 4),
            warmup_steps=self.config.get('warmup_steps', 100),
            logging_steps=self.config.get('logging_steps', 10),
            save_steps=self.config.get('steps', 500),
            save_strategy=("steps" if eval_dataset else "no"),
            eval_strategy=("steps" if eval_dataset else "no"),
            eval_steps=self.config.get('eval_steps', 500) if eval_dataset else None,
            save_total_limit=2,
            load_best_model_at_end=True if eval_dataset else False,
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4),
            learning_rate=self.config.get('learning_rate', 2e-4),
            fp16=False,  # Disable fp16 for Mac compatibility
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            max_grad_norm=0.3,
            group_by_length=True,
            report_to=None  # Disable wandb/tensorboard
        )
        
        # Data collator
        from transformers import DataCollatorForLanguageModeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        self.logger.info(f"Fine-tuning completed. Model saved to {self.output_dir}")
    
def main():
    """Example usage of the fine-tuning pipeline."""
    
    # Configuration
    config = {
        'base_model': 'facebook/bart-large-cnn',
        'output_dir': 'models/summarizer_lora',
        'num_epochs': 3,
        'batch_size': 2,
        'learning_rate': 2e-4,
        'max_length': 1024,
        'lora_r': 16,
        'lora_alpha': 32
    }
    
    # Initialize fine-tuner
    fine_tuner = AcademicSummarizationFineTuner(config)
    
    # Create sample dataset
    dataset_path = 'data/training/sample_academic_summaries.json'
    training_json_path = 'data/training/training_data.json'
    fine_tuner.create_sample_dataset(training_json_path, dataset_path)
    
    # Prepare model
    fine_tuner.prepare_model()
    
    # Prepare dataset
    train_dataset = fine_tuner.prepare_dataset(dataset_path)
    
    # Split for evaluation (80/20)
    train_size = int(0.8 * len(train_dataset))
    eval_size = len(train_dataset) - train_size
    
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, eval_size]
    )
    
    # Train
    fine_tuner.train(train_dataset, eval_dataset)

if __name__ == "__main__":
    main()
