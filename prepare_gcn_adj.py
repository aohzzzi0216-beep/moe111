# ==========================================
# 文件名: prepare_gcn_adj.py (Large 版本)
# ==========================================
import os
import json
import torch
import spacy
from transformers import AutoTokenizer
from tqdm import tqdm

# ==========================================
# 全局配置
# ==========================================
#  修改：与主干网络对齐，使用 large 版本的分词器
MODEL_NAME = "microsoft/deberta-v3-large"
GLOBAL_MAX_LEN = 512

# 加载 spacy 英文模型 (请确保已安装: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os

    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def generate_adj_matrices(jsonl_path, output_pt_path):
    """
    针对 DeBERTa-v3 优化的句法邻接矩阵生成脚本
    修复：区间重叠对齐、O(E) 边表加速、稀疏化存储
    """
    # 显式开启 use_fast=True 否则无法获取 offset_mapping
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    adj_matrices = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"正在处理数据: {jsonl_path} ...")
    for line in tqdm(lines):
        item = json.loads(line)
        text = item.get("text", "")

        # 1. SpaCy 依存句法分析
        doc = nlp(text)
        spacy_tokens = [t for t in doc]

        # 2. Transformer 分词并获取字符级坐标
        encoding = tokenizer(
            text,
            max_length=GLOBAL_MAX_LEN,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True
        )
        offsets = encoding["offset_mapping"]
        adj = torch.zeros((GLOBAL_MAX_LEN, GLOBAL_MAX_LEN))

        # 🌟 修复 1：鲁棒的子词到单词对齐 (区间交集法)
        subword_to_spacy = {}
        for i, (start, end) in enumerate(offsets):
            if start == end: continue  # 跳过 [CLS], [SEP], [PAD]

            for j, token in enumerate(spacy_tokens):
                # 判定子词区间与单词区间是否有交集
                if max(start, token.idx) < min(end, token.idx + len(token)):
                    subword_to_spacy[i] = j
                    break

        # 🌟 优化 2：建立反向索引 (spacy_idx -> [subword_indices])
        spacy_to_subwords = {}
        for sub_idx, sp_idx in subword_to_spacy.items():
            if sp_idx not in spacy_to_subwords:
                spacy_to_subwords[sp_idx] = []
            spacy_to_subwords[sp_idx].append(sub_idx)

        # 🌟 优化 3：基于依存边 (Edges) 构建矩阵 (复杂度从 N^2 降至 E)
        for token in spacy_tokens:
            curr_idx = token.i
            head_idx = token.head.i

            # 如果当前词及其父节点都在 512 截断范围内
            if curr_idx in spacy_to_subwords and head_idx in spacy_to_subwords:
                for i in spacy_to_subwords[curr_idx]:
                    for j in spacy_to_subwords[head_idx]:
                        adj[i, j] = 1
                        adj[j, i] = 1

        # 🌟 修复 4：强制补全自环，确保 GCN 稳定性
        for i in range(len(offsets)):
            adj[i, i] = 1

        # 转换为稀疏格式保存，极大降低显存和磁盘占用
        adj_matrices.append(adj.to_sparse())

    # 📊 密度监控
    all_adjs_dense = [a.to_dense() for a in adj_matrices[:100]]  # 抽样前100个看密度
    if all_adjs_dense:
        avg_density = torch.count_nonzero(torch.stack(all_adjs_dense)) / (100 * GLOBAL_MAX_LEN ** 2)
        print(f"📊 邻接矩阵平均密度: {avg_density:.4f}% (若 > 0.1% 则表示句法特征提取成功)")

    # 保存稀疏列表
    torch.save(adj_matrices, output_pt_path)
    print(f"✅ 成功保存稀疏矩阵列表 -> {output_pt_path}\n")


if __name__ == "__main__":
    # 清理旧数据，防止格式冲突
    files_to_clean = ["train_adj.pt", "val_adj.pt", "test_adj.pt"]
    for f in files_to_clean:
        if os.path.exists(f):
            os.remove(f)
            print(f"清理旧矩阵: {f}")

    generate_adj_matrices("train_gpt2.jsonl", "train_adj.pt")
    generate_adj_matrices("val_gpt2.jsonl", "val_adj.pt")
    generate_adj_matrices("test_gpt2.jsonl", "test_adj.pt")
