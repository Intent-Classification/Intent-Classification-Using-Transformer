import torch 
import torch.nn as nn


class LayerNorm(nn.Module):
    def  __init__(self,emb_dim):
        super().__init__()
        '''
            y = scale * xnorm + shift
        '''
        self.eps = 1e-5 
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
        
    def forward(self,x):
        mean = x.mean(dim = -1, keep_dim = True)
        var = x.var(dim = -1, keep_dim = True, unbiased = False)
        norm_x = (x-mean)/torch.sqrt(var+self.eps)
        return self.scale*norm_x + self.shift
        
class GELU(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        return 0.5* x *(1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * (x + 0.044715 * torch.pow(x,3))))
     
    
        
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"],4*cfg["emb_dim"]),
            GELU(),
            nn.Linear(4*cfg["emb_dim"], cfg["emb_dim"])
        )
        
    def forward(self,x):
        return self.layers(x)
        

class MultiHeadATtention(nn.Module):
    def __init__(self, d_in, d_out, dropout,num_heads,qkv_bias = True ): #qkv bias helps classifer to seperate features 
        super().__init__()
        assert d_out % num_heads == 0 
        self.d_out = d_out 
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        
        self.W_query = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in,d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias=qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x, attention_mask = None):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        queries = self.W_query(x).view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        values = self.W_value(x).view(b,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        
        attn_scores = queries @ keys.transpose(2,3) /self.head_dim ** 0.5
        
        attn_weights = torch.softmax(attn_scores, dim = -1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b,num_tokens,self.d_out)
        return self.out_proj(context_vec)
    
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.att = MultiHeadATtention(
            d_in = cfg["emb_dim"], 
            d_out = cfg ["emb_dim"],
            dropout = cfg["drop_rate"],
            num_heads = cfg ["n_heads"],
            qkv_bias = True,           
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop = nn.Dropout(cfg["drop_rate"])
        
    def forward(self,x, attention_mask = None):

        shortcut = x 
        x = self.norm1(x)
        x = self.att(x,attention_mask)
        x = self.drop(x)
        x = x + shortcut
        
        shortcut = x 
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x = x + shortcut 
        return x 

class IntentClassifier(nn.Module):
    def __init__(self,cfg, num_classes, pool = "mean"):
        super().__init__()
        self.pool = pool 
        self.input_drop = nn.Dropout(cfg["drop_rate"])
        
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg)for _ in range(cfg["n_layers"])
        ])
        
        self.classifier = nn.Sequential(
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(cfg["emb_dim"],cfg["emb_dim"]),
            GELU(),
            nn.Linear(cfg["emb_dim"],num_classes)
        )
    
    def forward(self,x,attention_mask = None):
        x = self.input_drop(x)
        
        for block in self.blocks:
            x  = block(x, attention_mask)
            
        x = x.mean(dim = 1)
        
        return self.classifier(x)
    
        
        