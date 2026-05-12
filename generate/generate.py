# 生成脚本，使用训练好的模型从头生成新的核苷酸序列
import random
import argparse
import torch
import os
from tqdm import tqdm
from transformers import GPT2Config, GPT2LMHeadModel
from Lighting_module import LitUTRPretrain
from nucleotide_tokenizer import NucleotideTokenizer
import pytorch_lightning as pl

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
 

def build_model_and_tokenizer(vocab_path, n_embeding, n_layer, n_head, n_ctx):
    tokenizer = NucleotideTokenizer(vocab_file=vocab_path)
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=n_ctx,
        n_ctx=n_ctx,
        n_embd=n_embeding,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=None,
        eos_token_id=tokenizer.sep_token_id,
        pad_token_id=tokenizer.pad_token_id
    )
    hf_model = GPT2LMHeadModel(config)
    return tokenizer, hf_model


def generate_with_prompt(lit_model, tokenizer, prompt_ids, min_length, max_length, num, top_k, top_p, temperature, repetition_penalty):
    input_ids = prompt_ids.to(lit_model.device)
    with torch.no_grad(): 
        out = lit_model.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.sep_token_id
        )
    dec = tokenizer.batch_decode(out, skip_special_tokens=True)
    clean = [s.replace(" ", "").strip() for s in dec]
    return clean


def write_fasta(seqs, path, line_width=80):
    with open(path, "w") as f:
        for i, seq in enumerate(seqs, start=1):
            f.write(f">seq_{i}\n")
            for j in range(0, len(seq), line_width):
                f.write(seq[j:j+line_width] + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckp", type=str, required=True, help="Lightning 训练得到的模型 checkpoint 路径 (.ckpt)")

    p.add_argument("--vocab_path", type=str, default='vocab.txt', help="核苷酸 tokenizer 使用的词表文件")
    p.add_argument('--n_embd', type=int, default=384, help="GPT2 模型的嵌入维度 n_embd")
    p.add_argument('--n_layer', type=int, default=8, help="Transformer 层数 n_layer")
    p.add_argument('--n_head', type=int, default=16, help="多头注意力 head 数 n_head")
    p.add_argument('--max_length', type=int, default=256, help="模型最大上下文长度 (n_ctx / n_positions)")
    p.add_argument("--top_k", type=int, default=50, help="top-k 采样截断的候选 token 数")
    p.add_argument("--top_p", type=float, default=0.9, help="nucleus (top-p) 采样累计概率阈值")
    p.add_argument("--temperature", type=float, default=1.0, help="采样温度，>1 更随机，<1 更保守")
    p.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚系数，>1 可降低重复片段")

    p.add_argument("--num_sample", type=int, default=3000, help="用于生成的样本数量上限 (从 test_file 中随机采样)")
    p.add_argument('--batch_size', type=int, default=256, help='生成时的 batch size，影响显存与速度')
    p.add_argument('--min_gen_length', type=int, default=100, help='生成序列的最小长度')
    p.add_argument('--max_gen_length', type=int, default=150, help='生成序列的最大长度')

    p.add_argument("--output_dir", type=str, default="output/ep74", help="生成结果保存目录")
    p.add_argument("--seed", type=int, default=42, help="随机数种子，保证可复现实验")
    args = p.parse_args()


    set_seed(int(args.max_gen_length *1e+6)+int(args.min_gen_length *1e+3) + args.seed)
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer, hf_model = build_model_and_tokenizer(
        args.vocab_path,
        args.n_embd,
        args.n_layer,
        args.n_head,
        args.max_length
    )
    lit_model = LitUTRPretrain.load_from_checkpoint(
        args.ckp,
        model=hf_model,
        tokenizer=tokenizer
    )
    lit_model.eval().cuda()
    print("Loaded model from checkpoint and set to eval mode.")

    # 从头生成; 以ATCG单个字符作为 prompt
    unique_seqs = set()
    pbar = tqdm(total=args.num_sample, desc="Generating unique sequences")
    prompts = ['A', 'T', 'C', 'G']
    
    while len(unique_seqs) < args.num_sample:
        current_batch_size = min(args.batch_size, args.num_sample - len(unique_seqs))
        
        # 随机选择 prompts
        batch_prompts = random.choices(prompts, k=current_batch_size)
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", add_special_tokens=False, padding=True)
        prompt_ids = inputs["input_ids"].to(lit_model.device)
        
        generated_seqs = generate_with_prompt(
            lit_model,
            tokenizer,
            prompt_ids,
            args.min_gen_length,
            args.max_gen_length,
            1,  # 每次生成一个序列
            args.top_k,
            args.top_p,
            args.temperature,
            args.repetition_penalty
        )
        
        initial_count = len(unique_seqs)
        unique_seqs.update(generated_seqs)
        pbar.update(len(unique_seqs) - initial_count)

    pbar.close()
    # rows = sorted(list(unique_seqs))
    rows = list(unique_seqs)

    # 输出为 FASTA 文件
    output_fasta = os.path.join(args.output_dir, f"{args.min_gen_length}_{args.max_gen_length}.fasta")
    write_fasta(rows, output_fasta)

if __name__ == "__main__":
    main()


