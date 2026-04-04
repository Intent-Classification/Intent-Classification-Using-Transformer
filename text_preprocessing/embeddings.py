import torch 
import torch.nn as nn
from transformers import BertModel
from dotenv import load_dotenv
from text_preprocessing.tokenization import Banking77Dataset
from torch.utils.data import DataLoader
import os 

load_dotenv()
hf_token = os.environ.get('HF_TOKEN')

class BertEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertModel.from_pretrained('bert-base-uncased', token=hf_token)
        self.model.to(self.device)
        self.model.eval()
    
    def get_embeddings(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        return outputs.last_hidden_state  # (B, seq_len, hidden)
    

def collect_embeddings(loader, embedder):
    """Run a DataLoader through the embedder and return (embeddings, labels)."""
    all_embeddings, all_labels, all_attention_mask = [], [], []
    for batch in loader:
        emb = embedder.get_embeddings(batch["input_ids"], batch["attention_mask"]) 
        all_embeddings.append(emb.cpu()) #
        all_attention_mask.append(batch["attention_mask"])#The attention mask here is used for computation to not let padding affect attention token, not to save in .pt file.
        all_labels.append(batch["label"])
    return torch.cat(all_embeddings, dim=0), torch.cat (all_attention_mask, dim = 0), torch.cat(all_labels, dim=0)


if __name__ == "__main__":
    SEED       = 42
    VAL_RATIO  = 0.15
    MAX_LENGTH = 64
    BATCH_SIZE = 32

    embedder = BertEmbedder()
    print(f"Generating embeddings on: {embedder.device}\n")

    # ── Train embeddings ───────────────────────────────────────────────────────
    train_dataset = Banking77Dataset(split="train", max_length=MAX_LENGTH)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_embeddings,all_attention_mask, all_labels = collect_embeddings(train_loader, embedder)
    print(f"Full train — embeddings: {all_embeddings.shape}, attention_mask : {all_attention_mask.shape}, labels: {all_labels.shape}")

    # ── Train / val split ──────────────────────────────────────────────────────
    n         = len(all_labels)
    val_size  = int(VAL_RATIO * n)
    train_size = n - val_size

    # Reproducible shuffle
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(SEED))

    train_idx = idx[val_size:]      # larger portion
    val_idx   = idx[:val_size]      # smaller portion

    torch.save(
        {"embeddings": all_embeddings[train_idx], 
        "attention_mask": all_attention_mask[train_idx],
        "labels": all_labels[train_idx]},
        "train_embeddings.pt"
    )
    print(f"Saved train_embeddings.pt  ({train_size} samples)")

    torch.save(
        {"embeddings": all_embeddings[val_idx],
        "attention_mask": all_attention_mask[val_idx],
         "labels": all_labels[val_idx]},
        "val_embeddings.pt"
    )
    print(f"Saved val_embeddings.pt    ({val_size} samples)")

    # ── Test embeddings  ────────────────────────────────────────────
    test_dataset = Banking77Dataset(split="test", max_length=MAX_LENGTH)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    test_embeddings, test_attention_mask, test_labels = collect_embeddings(test_loader, embedder)
    print(f"\nTest — embeddings: {test_embeddings.shape}, labels: {test_labels.shape}")

    torch.save(
        {"embeddings": test_embeddings, 
         "attention_mask": test_attention_mask,
         "labels": test_labels},
        "test_embeddings.pt"
    )
    print("Saved test_embeddings.pt")