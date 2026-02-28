from transformers import BertTokenizer 
from datasets import load_dataset
from torch.utils.data import Dataset
from dotenv import load_dotenv
import torch
import os

load_dotenv()
hf_token = os.environ.get('HF_TOKEN')

''' Dataset class to load Banking77 training data'''
class Banking77Dataset(Dataset):
    def __init__(self, split = "train", max_length = 64):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',token = hf_token)
        self.max_length = max_length
        
        raw = load_dataset('mteb/banking77')
        self.data = raw[split]
        
    def __len__(self):  
            return len(self.data)
        
    def __getitem__(self,idx):
            item = self.data[idx]
        
            encoding = self.tokenizer(
                item["text"],
                padding="max_length",
                truncation = True,
                max_length = self.max_length,
                return_tensors = "pt"
            )
            
            return {
            "input_ids":encoding['input_ids'].squeeze(0),
            "attention_mask":encoding['attention_mask'].squeeze(0),
            "label": torch.tensor(item["label"], dtype= torch.long)
            }
            
if __name__ == "__main__":
    train_dataset = Banking77Dataset(split="train",max_length=64)
    sample = train_dataset[0]
    print("Padded tokens:", sample["input_ids"])
    print("Attention Padding: ", sample["attention_mask"])
    
'''
Sample means row(0) here 
inside raw(0) there are "text" and "label"
train_dataset means it feeds Banking77Dataset with required parameters and return we get input ids,attention padding and label.

'''