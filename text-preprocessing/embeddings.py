import torch 
import torch.nn as nn
from transformers import BertModel
from dotenv import load_dotenv
from tokenization import Banking77Dataset
from torch.utils.data import DataLoader
import os 

load_dotenv()
hf_token = os.environ.get('HF_TOKEN')

class BertEmbedder:
    def __init__(self,device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained('bert-base-uncased', token = hf_token)
        self.model.to(self.device)
        self.model.eval() #Disable dropout and batch normalization, not training only loading embedding
    
    def get_embeddings(self,input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        cls_embeddings = outputs.last_hidden_state
        return cls_embeddings
    
if __name__== "__main__":
    train_dataset = Banking77Dataset(split = "train", max_length =64)
    train_loader = DataLoader(train_dataset,batch_size=32,shuffle = False)
    
    embedder = BertEmbedder()
    all_embeddings = []
    all_labels = []
    
    print(f"Generating Embeddings on: {embedder.device}")
    
    for batch_idx,batch in enumerate(train_loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]
        
        embeddings = embedder.get_embeddings(input_ids,attention_mask)
        
        all_embeddings.append(embeddings.to('cpu'))
        all_labels.append(labels)
        
        if batch_idx == 0:
            print(f"Batch 0 — embedding shape: {embeddings.shape}")
            print(f"Sample embedding (first 8 dims): {embeddings[0,0, :8]}")
            
    all_embeddings = torch.cat(all_embeddings,dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"\nFinal embeddings shape: {all_embeddings.shape}")
    print(f"Final labels shape:     {all_labels.shape}")
    
    ''' Saving for classifier'''    
    
    torch.save({"embeddings":all_embeddings, "labels":all_labels}, "train_embeddings.pt")
    print("Saved to train_embeddings.pt")
    
    test_dataset = Banking77Dataset(split="test", max_length=64)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    all_test_embeddings = []
    all_test_labels = []

    for batch in test_loader:
        embeddings = embedder.get_embeddings(batch["input_ids"], batch["attention_mask"])
        all_test_embeddings.append(embeddings.to('cpu'))
        all_test_labels.append(batch["label"])

    all_test_embeddings = torch.cat(all_test_embeddings, dim=0)
    all_test_labels = torch.cat(all_test_labels, dim=0)

    torch.save({"embeddings": all_test_embeddings, "labels": all_test_labels}, "test_embeddings.pt")
    print("Saved to test_embeddings.pt")