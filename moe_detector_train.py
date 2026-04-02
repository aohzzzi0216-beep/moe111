# ==========================================
# 文件名: moe_detector_train.py
# ==========================================
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import random
import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 全局参数
GLOBAL_MAX_LEN = 512
THRESHOLD = 128  # 长短文本判别阈值


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


## ==========================================
#  1. 强鲁棒性的图卷积网络 (防 NaN 爆炸)
## ==========================================
class GCNLayer(nn.Module):
    # (保留原有 GCNLayer 不变)
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, text_features, adj_matrix):
        support = torch.matmul(text_features, self.weight)
        adj_matrix = adj_matrix.to(support.dtype)
        degree = adj_matrix.sum(dim=-1, keepdim=True)
        degree = torch.clamp(degree, min=1e-9)
        norm_adj_matrix = adj_matrix / degree
        output = torch.bmm(norm_adj_matrix, support)
        return F.relu(output)

# 共享的深层句法特征底座 (只负责提取全局拓扑，不参与分支路由)
class SharedSyntaxGCNBase(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states, adj_matrix):
        x = self.gcn1(hidden_states, adj_matrix)
        x = self.gcn2(x, adj_matrix)
        return self.layer_norm(x + hidden_states) # 输出统一的 H_syn


## ==========================================
# 2. 细粒度数据集 (Token 级别 0/1 对齐)
## ==========================================
class SeqXGPTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512, adj_matrix_path=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.adj_matrices = None

        if adj_matrix_path and os.path.exists(adj_matrix_path):
            print(f"加载离线 GCN 邻接矩阵: {adj_matrix_path}")
            self.adj_matrices = torch.load(adj_matrix_path)

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            #  引入枚举，追踪每一行的原始行号
            for line_idx, line in enumerate(f):
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                except:
                    continue
                text = item.get("text", "").strip()
                prompt_len = item.get("prompt_len", 0)


                # 过滤逻辑
                # 替换为你第一版的逻辑
                if prompt_len <= 0 or not text:
                    continue

                self.data.append({
                    "text": text,
                    "prompt_len": prompt_len,
                    "orig_line_idx": line_idx  #  记录原始行号
                })

        # 增加对齐校验
        if self.adj_matrices is not None:
            max_orig_idx = max([d["orig_line_idx"] for d in self.data]) if self.data else 0
            if max_orig_idx >= len(self.adj_matrices):
                raise ValueError(
                    f"❌ 矩阵文件长度({len(self.adj_matrices)}) 与 数据集原始行数({max_orig_idx + 1}) 不匹配！请重新运行预处理脚本。")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        prompt_len = item["prompt_len"]
        orig_line_idx = item["orig_line_idx"]

        # 🌟 修复 2：抛弃词级对齐，改用绝对字符边界对齐
        words = text.split()
        # 还原 prompt 部分的原始文本，并计算其准确的字符长度
        prompt_text = " ".join(words[:prompt_len])
        prompt_char_boundary = len(prompt_text)

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        offsets = encoding["offset_mapping"].squeeze(0)  # [seq_len, 2]

        token_labels = []
        for start_char, end_char in offsets:
            if start_char == end_char:
                token_labels.append(-100)
            elif end_char <= prompt_char_boundary:
                token_labels.append(0)
            elif start_char >= prompt_char_boundary:
                token_labels.append(1)
            else:
                token_labels.append(1)

        seq_len = min(len(words), self.max_length)

        # ========== 修复后的三元组提取逻辑 ==========
        human_text = text[:prompt_char_boundary].strip()
        ai_text = text[prompt_char_boundary:].strip()

        # 安全截取逻辑
        text_anc = human_text[-min(len(human_text), 800):]
        text_pos = ai_text[:min(len(ai_text), 800)]
        h_tail = human_text[-min(len(human_text), 400):]
        a_head = ai_text[:min(len(ai_text), 400)]
        text_neg = h_tail + " " + a_head

        # 空格兜底
        t_anc = text_anc if text_anc else " "
        t_pos = text_pos if text_pos else " "
        t_neg = text_neg if text_neg else " "

        # 统一 Tokenize
        enc_anc = self.tokenizer(t_anc, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        enc_pos = self.tokenizer(t_pos, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        enc_neg = self.tokenizer(t_neg, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(token_labels, dtype=torch.long),
            "seq_length": torch.tensor(seq_len, dtype=torch.long),
            "anc_ids": enc_anc["input_ids"].squeeze(0),
            "anc_mask": enc_anc["attention_mask"].squeeze(0),
            "pos_ids": enc_pos["input_ids"].squeeze(0),
            "pos_mask": enc_pos["attention_mask"].squeeze(0),
            "neg_ids": enc_neg["input_ids"].squeeze(0),
            "neg_mask": enc_neg["attention_mask"].squeeze(0)
        }

        # 邻接矩阵加载逻辑
        if self.adj_matrices is not None:
            try:
                adj_sparse = self.adj_matrices[orig_line_idx]
                result['adj_matrix'] = adj_sparse.to_dense().to(torch.float32)
            except IndexError:
                print(f"⚠️ Warning: 找不到行号 {orig_line_idx} 对应的矩阵，已使用单位矩阵代替。")
                result['adj_matrix'] = torch.eye(self.max_length, dtype=torch.float32)
        else:
            result['adj_matrix'] = torch.eye(self.max_length, dtype=torch.float32)

        return result

## ==========================================
# 3. 细粒度 Token 级分簇 MoE 检测器
## ==========================================
class MoEDetector(nn.Module):
    def __init__(self, backbone_name="microsoft/deberta-v2-xlarge", num_sem_experts=3, num_syn_experts=3):
        super().__init__()
        self.backbones = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.backbones.config.hidden_size

        # 全局路由器
        self.router = nn.Linear(hidden_size, 8)

        # 实例化唯一的共享句法底座
        self.shared_syntax_base = SharedSyntaxGCNBase(hidden_size)

        # 句法专家簇现在为接受 H_syn 并进行功能分化的轻量级网络
        self.syn_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1))
            for _ in range(num_syn_experts)
        ])


        # 2. 长度专家簇
        self.len_expert_short = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1))
        self.len_expert_long = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1))

        # 3. 语义专家簇
        self.sem_experts = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(0.1))
            for _ in range(num_sem_experts)
        ])

        # 最终分类器
        self.classifier = nn.Linear(hidden_size, 2)

    def get_semantic_embedding(self, input_ids, attention_mask):
        """专用于对比学习：独立提取片段的语义专家联合表征"""
        outputs = self.backbones(input_ids=input_ids, attention_mask=attention_mask)
        # 取 [CLS] token 代表这 128 个 Token 窗口的全局语义
        cls_state = outputs.last_hidden_state[:, 0, :]

        # 汇总所有语义专家的处理结果，取平均
        sem_embs = [expert(cls_state) for expert in self.sem_experts]
        return torch.stack(sem_embs, dim=0).mean(dim=0)

    def forward(self, input_ids, attention_mask, seq_lengths, adj_matrix=None, labels=None,
                anc_ids=None, anc_mask=None, pos_ids=None, pos_mask=None, neg_ids=None, neg_mask=None):
        outputs = self.backbones(input_ids=input_ids, attention_mask=attention_mask)
        # 获取所有 Token 的特征 [batch, seq_len, hidden_size]
        hidden_states = outputs.last_hidden_state

        # 1. 计算所有 Token 的全局 8 专家原始 Logits
        router_logits = self.router(hidden_states)  # [batch, seq_len, 8]

        # 2. 长度专家强制屏蔽机制
        is_short = (seq_lengths <= THRESHOLD).view(-1, 1).unsqueeze(-1)  # [batch, 1, 1]
        masked_router_logits = router_logits.clone()

        masked_router_logits[:, :, 4] = masked_router_logits[:, :, 4].masked_fill(is_short.squeeze(-1), -1e9)
        masked_router_logits[:, :, 3] = masked_router_logits[:, :, 3].masked_fill(~is_short.squeeze(-1), -1e9)

        # 3. 基础概率分布 W
        router_probs = F.softmax(masked_router_logits, dim=-1)  # [batch, seq_len, 8]

        # 4. 各专家簇独立竞选 Top-1
        syn_prob, syn_idx = torch.max(router_probs[:, :, 0:3], dim=-1)  # [batch, seq_len]
        len_prob, len_idx_local = torch.max(router_probs[:, :, 3:5], dim=-1)
        len_idx = len_idx_local + 3
        sem_prob, sem_idx_local = torch.max(router_probs[:, :, 5:8], dim=-1)
        sem_idx = sem_idx_local + 5

        # 5. Top-3 稀疏选择与重归一化
        top3_probs = torch.stack([syn_prob, len_prob, sem_prob], dim=-1)  # [batch, seq_len, 3]
        top3_probs = top3_probs / top3_probs.sum(dim=-1, keepdim=True)

        fused_features = torch.zeros_like(hidden_states)

        # 所有 Token 先统一通过共享 GCN 底座，提取出深层句法特征 H_syn
        shared_syntax_features = self.shared_syntax_base(hidden_states, adj_matrix)

        # -- 聚合句法专家 --
        for i, expert in enumerate(self.syn_experts):
            # 每个专家接受统一的句法表征，进行内部的细分功能推断
            expert_out = expert(shared_syntax_features)
            mask = (syn_idx == i)  # [batch, seq_len]
            fused_features[mask] += expert_out[mask] * top3_probs[..., 0][mask].unsqueeze(-1)


        # -- 聚合长度专家 --
        short_out = self.len_expert_short(hidden_states)
        mask_short = (len_idx == 3)
        fused_features[mask_short] += short_out[mask_short] * top3_probs[..., 1][mask_short].unsqueeze(-1)

        long_out = self.len_expert_long(hidden_states)
        mask_long = (len_idx == 4)
        fused_features[mask_long] += long_out[mask_long] * top3_probs[..., 1][mask_long].unsqueeze(-1)

        # -- 聚合语义专家 --
        for i, expert in enumerate(self.sem_experts):
            expert_out = expert(hidden_states)
            mask = (sem_idx == i + 5)
            fused_features[mask] += expert_out[mask] * top3_probs[..., 2][mask].unsqueeze(-1)

        # 细粒度逐词分类
        logits = self.classifier(fused_features)  # [batch, seq_len, 2]

        # 初始化 Loss
        loss = None

        # 只有在传入标签时（训练/验证阶段）计算 Loss
        if labels is not None:
            # 1. 计算主分类任务的交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # 展平进行计算，忽略 Padding 的部分
            active_logits = logits.view(-1, 2)
            active_labels = labels.view(-1)
            cls_loss = loss_fct(active_logits, active_labels)

            # 2. 计算标准负载均衡损失 (防止专家坍缩)
            def calc_balance_loss(probs):
                # probs: [batch, seq_len, num_experts]
                mean_probs = probs.mean(dim=(0, 1))  # 求得每个专家的平均被选中概率
                num_experts = mean_probs.size(0)
                # 均方惩罚：当所有专家概率均为 1/num_experts 时取得极小值，强制负载均衡
                return num_experts * torch.sum(mean_probs * mean_probs)

            balance_loss = calc_balance_loss(router_probs[:, :, 0:3]) + \
                           calc_balance_loss(router_probs[:, :, 5:8])

            # 合并基础损失
            loss = cls_loss + 0.01 * balance_loss

            # 3. ============ 核心模块：引入语义专家的对比学习损失 ============
            # 只有在训练阶段（传入了三元组数据）才累加对比损失
            if all(v is not None for v in [anc_ids, pos_ids, neg_ids]):
                # 🌟 增加校验：确保当前 Batch 的 Mask 有效，防止对纯 Padding 计算导致 NaN
                if anc_mask.sum() > 0 and pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    emb_anc = self.get_semantic_embedding(anc_ids, anc_mask)
                    emb_pos = self.get_semantic_embedding(pos_ids, pos_mask)
                    emb_neg = self.get_semantic_embedding(neg_ids, neg_mask)

                    # Margin 设置为 1.0，使用欧式距离 (p=2)
                    triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
                    contrastive_loss = triplet_loss_fn(emb_anc, emb_pos, emb_neg)

                    # 以 0.1 的权重汇入总 Loss
                    loss = loss + 0.1 * contrastive_loss
            # ========================================================

        return loss, logits


## ==========================================
# 评估逻辑更新 (过滤 Padding 进行验证)
## ==========================================
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lengths = batch['seq_length'].to(device)
            labels = batch['labels'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)

            _, logits = model(input_ids, attention_mask, seq_lengths, adj_matrix)
            preds = torch.argmax(logits, dim=-1).view(-1)
            labels = labels.view(-1)

            # 去除 padding 部分
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    p = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    r = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return p, r, f1


if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ 当前使用的计算设备: {device}")

    print("正在加载 deberta-v2-xlarge 和 模型...")
    # DeBERTa-V2 分词器直接加载即可
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge", use_fast=True)

    print("正在加载已物理切分的训练、验证、测试集及 GCN 邻接矩阵...")

    train_dataset = SeqXGPTDataset("train_gpt2.jsonl", tokenizer, max_length=GLOBAL_MAX_LEN,
                                   adj_matrix_path="train_adj.pt")
    val_dataset = SeqXGPTDataset("val_gpt2.jsonl", tokenizer, max_length=GLOBAL_MAX_LEN, adj_matrix_path="val_adj.pt")
    test_dataset = SeqXGPTDataset("test_gpt2.jsonl", tokenizer, max_length=GLOBAL_MAX_LEN,
                                  adj_matrix_path="test_adj.pt")

    g = torch.Generator()
    g.manual_seed(42)

    #显存保护：恢复 batch_size = 2
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    print(f"✅ 数据加载完成：训练集 {len(train_dataset)} | 验证集 {len(val_dataset)} | 测试集 {len(test_dataset)}")

    model = MoEDetector(backbone_name="microsoft/deberta-v2-xlarge", num_sem_experts=3, num_syn_experts=3)
    model.backbones.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # 定义不需要权重衰减的参数类型
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        # --- 第一部分：专家簇、Router、GCN (快速学习) ---
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbones" not in n and not any(nd in n for nd in no_decay)
            ],
            "lr": 1e-4,
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if "backbones" not in n and any(nd in n for nd in no_decay)
            ],
            "lr": 1e-4,
            "weight_decay": 0.0,
        },

        # --- 第二部分：DeBERTa 底座 (温和微调) ---
        {
            "params": [
                p for n, p in model.backbones.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "lr": 1e-5,  # 比专家簇低一个数量级
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in model.backbones.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "lr": 1e-5,
            "weight_decay": 0.0,
        },
    ]
    # 正式创建 optimizer 实例
    # 将参数组传给 AdamW，这样优化器就知道不同层该用什么学习率和权重衰减了
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    # 正确方式：先计算每轮 Epoch 实际触发多少次 optimizer.step()
    epochs = 20
    accumulation_steps = 16

    # 使用向上取整除法，确保最后不满 accumulation_steps 的那一个 batch 也能被计入更新步数
    steps_per_epoch = (len(train_dataloader) + accumulation_steps - 1) // accumulation_steps
    total_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    best_val_f1 = 0
    patience_counter = 0
    patience_limit = 3

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()  # 移动至循环开端，配合梯度累加

        for step, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lengths = batch['seq_length'].to(device)
            labels = batch['labels'].to(device)
            adj_matrix = batch['adj_matrix'].to(device)

            # --- 提取对比学习三元组并送入 device ---
            anc_ids = batch.get('anc_ids').to(device) if 'anc_ids' in batch else None
            anc_mask = batch.get('anc_mask').to(device) if 'anc_mask' in batch else None
            pos_ids = batch.get('pos_ids').to(device) if 'pos_ids' in batch else None
            pos_mask = batch.get('pos_mask').to(device) if 'pos_mask' in batch else None
            neg_ids = batch.get('neg_ids').to(device) if 'neg_ids' in batch else None
            neg_mask = batch.get('neg_mask').to(device) if 'neg_mask' in batch else None

            # 送入模型
            loss, _ = model(input_ids, attention_mask, seq_lengths, adj_matrix, labels,
                            anc_ids, anc_mask, pos_ids, pos_mask, neg_ids, neg_mask)

            loss = loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps

            if step % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}] | Step [{step}/{len(train_dataloader)}] | Loss: {loss.item() * accumulation_steps:.4f}")

        # Validation
        val_p, val_r, val_macro_f1 = evaluate(model, val_dataloader, device)
        print("-" * 40)
        print(f"【Epoch {epoch + 1} 验证集(Val)结果】 F1: {val_macro_f1 * 100:.2f}%")

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_moe_model.pth")
            print(f"验证集分更高！已保存当前最佳模型。")
        else:
            patience_counter += 1
            print(f"⚠️ 验证集未提升 (早停计数: {patience_counter}/{patience_limit})")
            if patience_counter >= patience_limit:
                print("触发早停机制，停止训练！")
                break
        print("-" * 40 + "\n")

    print("=" * 50)
    print("正在加载最佳模型，在完全未见过的 Test 测试集上进行最终评估...")
    model.load_state_dict(torch.load("best_moe_model.pth"))

    test_p, test_r, test_macro_f1 = evaluate(model, test_dataloader, device)

    print(f"🎉 最终 Test 测试集盲测成绩 :")
    print(f"Precision: {test_p * 100:.2f}% | Recall: {test_r * 100:.2f}% | Macro-F1: {test_macro_f1 * 100:.2f}%")
    print("=" * 50)

    log_filename = "experiment_logs.txt"
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"[{current_time}] 测试集评估结果 \n")
        f.write(f"  - Precision: {test_p * 100:.2f}%\n")
        f.write(f"  - Recall:    {test_r * 100:.2f}%\n")
        f.write(f"  - Macro-F1:  {test_macro_f1 * 100:.2f}%\n")
        f.write(
            f"  - 核心参数: max_len={GLOBAL_MAX_LEN}, epochs={epochs}, effective_batch_size=32 (8x4), backbone=microsoft/deberta-v2-xlarge\n")
        f.write("-" * 50 + "\n")