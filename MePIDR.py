# %%
import os
import argparse
import torch
import numpy as np
import random

# %%
parser = argparse.ArgumentParser()

parser.add_argument('--model_name', default='MePIDR')
parser.add_argument('--dataset', default='') # weibo/Pheme/Politifact
parser.add_argument('--epoch', type=int, default=25)
parser.add_argument('--max_len', type=int, default=197)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=4)
parser.add_argument('--bert_vocab_file', default='')
parser.add_argument('--root_path', default='')
parser.add_argument('--bert', default='')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3035)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='')
parser.add_argument('--save_param_dir', default= '')

parser.add_argument('--fc5_in_features', type=int, default=4096, help='Number of input features for fc5')
parser.add_argument('--fc4_in_features', type=int, default=2048, help='Number of input features for fc4')
parser.add_argument('--fc3_in_features', type=int, default=1024, help='Number of output features for fc3')
parser.add_argument('--fc2_in_features', type=int, default=512, help='Number of input features for fc2')
parser.add_argument('--fc1_in_features', type=int, default=300, help='Number of input features for fc1')

args = parser.parse_args([])

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# %%
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True

if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

# %%
config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir,
        'dataset':args.dataset,
        'dropout' : 0.6,
        'n_heads' : 8,
        'num_classes' : 2,
        'target_names' :['NR', 'FR']
        }

# %%
import os
import tqdm
import torch
import torch.nn as nn
import models_mae
import cn_clip.clip as clip
import torch.nn.init as init

# %%
# Transformer with memory slots
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):    
    def __init__(self, 
                 input_size: int, 
                 d_k: int = 16, 
                 d_v: int = 16, 
                 n_heads: int = 8, 
                 is_layer_norm: bool = False, 
                 attn_dropout: float = 0.1,
                 ffn_dropout: float = 0.1,
                 num_memory_slots: int = 4): # or 6
        super(TransformerBlock, self).__init__()
        
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v
        self.input_size = input_size
        self.is_layer_norm = is_layer_norm
        self.num_memory_slots = num_memory_slots
        
        # Initialize
        if is_layer_norm:
            self.layer_norm1 = nn.LayerNorm(input_size)
            self.layer_norm2 = nn.LayerNorm(input_size)
        
        self.W_q = nn.Linear(input_size, n_heads * d_k)
        self.W_k = nn.Linear(input_size, n_heads * d_k)
        self.W_v = nn.Linear(input_size, n_heads * d_v)
        self.W_o = nn.Linear(d_v * n_heads, input_size)
        
        self.memory_k = nn.Parameter(torch.Tensor(num_memory_slots, n_heads * d_k))
        self.memory_v = nn.Parameter(torch.Tensor(num_memory_slots, n_heads * d_v))
        
        self.ffn = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(input_size, input_size)
        )
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.ffn_dropout = nn.Dropout(ffn_dropout)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
        for layer in [self.ffn[0], self.ffn[3]]:
            init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        init.xavier_normal_(self.memory_k)
        init.xavier_normal_(self.memory_v)
    
    def scaled_dot_product_attention(self, 
                                   Q: torch.Tensor, 
                                   K: torch.Tensor, 
                                   V: torch.Tensor) -> torch.Tensor:

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        output = torch.matmul(attn_probs, V)
        return output
    
    def multi_head_attention(self, 
                           Q: torch.Tensor, 
                           K: torch.Tensor, 
                           V: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len, _ = Q.size()
        
        Q_proj = self.W_q(Q).view(batch_size, seq_len, self.n_heads, self.d_k)
        K_proj = self.W_k(K).view(batch_size, seq_len, self.n_heads, self.d_k)
        V_proj = self.W_v(V).view(batch_size, seq_len, self.n_heads, self.d_v)
        
        # Memory slots
        memory_k = self.memory_k.unsqueeze(0).expand(batch_size, -1, -1)
        memory_v = self.memory_v.unsqueeze(0).expand(batch_size, -1, -1)
        
        memory_k = memory_k.view(batch_size, self.num_memory_slots, self.n_heads, self.d_k)
        memory_v = memory_v.view(batch_size, self.num_memory_slots, self.n_heads, self.d_v)
        
        K_proj = torch.cat([K_proj, memory_k], dim=1)
        V_proj = torch.cat([V_proj, memory_v], dim=1)
        
        Q_proj = Q_proj.transpose(1, 2).reshape(batch_size * self.n_heads, seq_len, self.d_k)
        K_proj = K_proj.transpose(1, 2).reshape(batch_size * self.n_heads, seq_len + self.num_memory_slots, self.d_k)
        V_proj = V_proj.transpose(1, 2).reshape(batch_size * self.n_heads, seq_len + self.num_memory_slots, self.d_v)
        
        attn_output = self.scaled_dot_product_attention(Q_proj, K_proj, V_proj)
        
        attn_output = attn_output.view(batch_size, self.n_heads, seq_len, self.d_v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.W_o(attn_output)
        
        return output
    
    def forward(self, 
               Q: torch.Tensor, 
               K: Optional[torch.Tensor] = None, 
               V: Optional[torch.Tensor] = None) -> torch.Tensor:

        if K is None:
            K = Q
        if V is None:
            V = Q
        
        attn_output = self.multi_head_attention(Q, K, V)
        if self.is_layer_norm:
            X = self.layer_norm1(Q + attn_output)
        else:
            X = Q + attn_output
        
        ffn_output = self.ffn(X)
        if self.is_layer_norm:
            output = self.layer_norm2(X + ffn_output)
        else:
            output = X + ffn_output
        
        return output

# %%
class DynamicFeatureRouting(nn.Module):
    def __init__(self, feature_dim):
        super(DynamicFeatureRouting, self).__init__()
        self.feature_dim = feature_dim
        self.router = nn.Sequential(
            nn.Linear(feature_dim * 4, feature_dim * 2),
            nn.ReLU(),
            nn.Linear(feature_dim * 2, 4),
            nn.Softmax(dim=1)
        )

    def forward(self, features):
        concat_features = torch.cat(features, dim=1)
        routing_weights = self.router(concat_features)
        avg_routing_weights = routing_weights.mean(dim=0)
        routed_features = []
        for i, feat in enumerate(features):
            weight = routing_weights[:, i].unsqueeze(1)
            routed_feat = feat * weight
            routed_features.append(routed_feat)
            
        return torch.cat(routed_features, dim=1), avg_routing_weights

# %%
import torch.nn.init as init

class MePIDRModel(nn.Module):
    def __init__(self, bert):
        super(MePIDRModel, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        # Image model setup
        self.model_size = "base"
        self.image_model = models_mae.__dict__["mae".format(self.model_size)](norm_pix_loss=False)
        checkpoint = torch.load('.pth'.format(self.model_size), map_location='cpu')
        self.image_model.load_state_dict(checkpoint['model'], strict=False)
        
        # CLIP model setup
        self.ClipModel, _ = load_from_name("ViT-B-16", device="cuda", download_root='./')

        self.mh_attention = TransformerBlock(input_size=1024, n_heads=8, attn_dropout=0)

        # Fusion modules for MB and CLIP features
        self.bd2cd = nn.Sequential(
            nn.Linear(1536, 1024), 
        )

        dropout_rate = 0.6
        self.dropout = nn.Dropout(dropout_rate)

        # Dynamic Feature Routing Module
        self.dynamic_router = DynamicFeatureRouting(feature_dim=1024)

        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(args.fc5_in_features, args.fc4_in_features)#
        self.fc4 = nn.Linear(args.fc4_in_features, args.fc3_in_features)#
        self.fc3 = nn.Linear(args.fc3_in_features, args.fc2_in_features)#
        self.fc2 = nn.Linear(args.fc2_in_features, args.fc1_in_features)
        self.fc1 = nn.Linear(in_features=args.fc1_in_features, out_features=config['num_classes'])
        self.init_weight()   

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc3.weight)
        init.xavier_normal_(self.fc4.weight)
        init.xavier_normal_(self.fc5.weight)


    def forward(self, **kwargs):
        # Feature extraction
        inputs = kwargs['content']
  
        masks = kwargs['content_masks']

        bert_text_feature = self.bert(inputs, attention_mask=masks)[0]

        image = kwargs['image']

        mae_image_feature = self.image_model.forward_ying(image)

        clip_image = kwargs['clip_image']

        clip_text = kwargs['clip_text']
        
        # CLIP features
        with torch.no_grad():
            clip_img_features = self.ClipModel.encode_image(kwargs['clip_image'])
            clip_txt_features = self.ClipModel.encode_text(kwargs['clip_text'])
            clip_img_features = clip_img_features / clip_img_features.norm(dim=-1, keepdim=True)
            clip_txt_features = clip_txt_features / clip_txt_features.norm(dim=-1, keepdim=True)

        # Fusion
        fusion_mb = torch.cat([mae_image_feature.mean(dim=1), bert_text_feature.mean(dim=1)], dim=-1)

        fusion_clip = torch.cat([clip_img_features, clip_txt_features], dim=-1).float()

        text_feature = self.bd2cd(fusion_mb)

        image_feature = fusion_clip
 
        att_len = 1024
        bsz = text_feature.size()[0]
        
        self_att_t = self.mh_attention(text_feature.view(bsz, -1, att_len), text_feature.view(bsz, -1, att_len), \
                                       text_feature.view(bsz, -1, att_len))

        self_att_i = self.mh_attention(image_feature.view(bsz, -1, att_len), image_feature.view(bsz, -1, att_len), \
                                       image_feature.view(bsz, -1, att_len))

        self_i = self_att_i.view(bsz, att_len)
        self_t = self_att_t.view(bsz, att_len)

        text_enhanced = self.mh_attention(self_att_i.view((bsz, -1, att_len)), self_att_t.view((bsz, -1, att_len)), \
                                          self_att_t.view((bsz, -1, att_len))).view(bsz, att_len)

        self_att_t = text_enhanced.view((bsz, -1, att_len))

        co_att_ti = self.mh_attention(self_att_t, self_att_i, self_att_i).view(bsz, att_len)
        co_att_it = self.mh_attention(self_att_i, self_att_t, self_att_t).view(bsz, att_len)

        features_to_route = [self_i, co_att_it, co_att_ti, self_t]
        routed_features, avg_routing_weights = self.dynamic_router(features_to_route)

        a1 = self.relu(self.dropout(self.fc5(routed_features)))
        a1 = self.relu(self.fc4(a1))
        a1 = self.relu(self.fc3(a1))
        a1 = self.relu(self.fc2(a1))
        d1 = self.dropout(a1)
        output = self.fc1(d1)

        return output, avg_routing_weights

# %%
from sklearn.metrics import classification_report, accuracy_score

class MePIDRTrainer():
    def __init__(
        self,        
        bert_model_name,
        use_cuda,
        lr,
        dropout,
        train_loader,
        val_loader,
        test_loader,
        weight_decay,
        early_stop=4,
        epochs=25
    ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.early_stop = early_stop
        self.epochs = epochs
        self.use_cuda = use_cuda
        self.bert_model_name = bert_model_name
        self.dropout = dropout
        self.best_acc = 0
        self.init_clip_max_norm = None

    def train(self):
        self.model = MePIDRModel(self.bert_model_name)
        
        if self.use_cuda:
            self.model = self.model.cuda()

        loss = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW( self.model.parameters(), lr= self.lr)
        
        for epoch in range(self.epochs):
            print("\nEpoch ", epoch + 1, "/", self.epochs)
            self.model.train()
            data_iter = tqdm.tqdm(self.train_loader, desc="Training", leave=False)
            for step_n, batch in enumerate(data_iter):
                total = len(data_iter)
                batch_data = clipdata2gpu(batch)
                labels = batch_data['label']
                logit_defense, _ = self.model(**batch_data)
                loss_classification = loss(logit_defense, labels.long())
                loss_defense = loss_classification
                self.optimizer.zero_grad()
                loss_defense.backward()
                self.optimizer.step()
                corrects = (torch.max(logit_defense, 1)[1].view(labels.size()).data == labels.data).sum()
                accuracy = 100 * corrects / len(labels)
            self.evaluate()

    def evaluate(self):
        y_pred = []
        y_dev = []

        self.model.eval()
        data_iter = tqdm.tqdm(self.test_loader, desc="Test", leave=False)
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = clipdata2gpu(batch)
                labels = batch_data['label']
                y_dev += labels.data.cpu().numpy().tolist()
                logits, avg_routing_weights = self.model(**batch_data)
                predicted = torch.max(logits, dim=1)[1]
                y_pred += predicted.data.cpu().numpy().tolist()
        acc = accuracy_score(y_dev, y_pred)
        
        if acc > self.best_acc:
            self.best_acc = acc
            print(classification_report(y_dev, y_pred,  digits=5))
            print("Best val set acc:", self.best_acc)
                                
# %%
class Run():
    def __init__(self,config):
        
        self.configinfo = config # Input parameters

        self.use_cuda = config['use_cuda']
        self.model_name = config['model_name']
        self.lr = config['lr']
        self.batchsize = config['batchsize']
        self.emb_type = config['emb_type']
        self.emb_dim = config['emb_dim']
        self.max_len = config['max_len']
        self.num_workers = config['num_workers']
        self.vocab_file = config['vocab_file']
        self.early_stop = config['early_stop']
        self.bert = config['bert']
        self.root_path = config['root_path']
        self.dropout = config['model']['dropout']
        self.seed = config['seed']
        self.weight_decay = config['weight_decay']
        self.epoch = config['epoch']
        self.save_param_dir = config['save_param_dir']
        self.dataset = config['dataset']
        
        if config['dataset']=="":
            self.root_path = ''
            self.train_path = self.root_path + '.csv'
            self.val_path = self.root_path + '.csv'
            self.test_path = self.root_path + '.csv' 

    def main(self):
        train_loader, val_loader, test_loader = self.get_dataloader(self.dataset)
        trainer = MePIDRTrainer( bert_model_name=self.bert,use_cuda=self.use_cuda, lr=self.lr, 
                                train_loader=train_loader, dropout=self.dropout,
                                weight_decay=self.weight_decay, val_loader=val_loader,
                                test_loader=test_loader, 
                                early_stop=self.early_stop, epochs=self.epoch,
                                )
        trainer.train()

# %%

if __name__ == '__main__':
    Run(config = config).main()


