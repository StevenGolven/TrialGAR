# -*- coding: utf-8 -*-


import os
import re
import json
import time
import pickle
import argparse
from typing import List, Dict, Any, Tuple
from datetime import datetime

import jieba
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ================= 配置与常量 =================

# 上下文管理常量（针对 DeepSeek 的 token 限制）
MAX_CONTEXT_TOKENS = 4096  # DeepSeek context window
CRITERIA_HEAD_KEEP = 8     # 保留前 N 条（核心疾病定义）
CRITERIA_TAIL_KEEP = 4     # 保留后 N 条（关键禁忌症、孕妇、过敏等）

# 中文停用词（用于 BM25）
STOPWORDS = set([
    "的", "了", "和", "或", "与", "及", "以及", "等",
    "研究", "试验", "临床", "患者", "方案", "治疗", "评估", "观察",
    "随机", "对照", "多中心", "入组", "排除", "纳入", "受试者",
    "符合", "满足", "具有", "进行", "接受", "完成",
])

# ================= 工具函数 =================

def normalize_text(s: str) -> str:
    """文本标准化"""
    if not s:
        return ""
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_zh(text: str) -> List[str]:
    """
    中文分词（用于 BM25）
    - jieba 分词
    - 去停用词
    - 去短词（< 2字符）
    """
    text = normalize_text(text)
    if not text:
        return []
    
    tokens = jieba.lcut(text)
    result = []
    
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if token in STOPWORDS:
            continue
        if len(token) < 2:
            continue
        result.append(token)
    
    return result

# ================= 数据加载与索引 =================

def load_trials(data_dir: str, max_trials: int = 0) -> List[Dict[str, Any]]:
    """
    加载清洗后的 trial 数据
    
    Args:
        data_dir: 包含 *.jsonl 文件的目录
        max_trials: 最多加载数量（0 = 全部）
    
    Returns:
        List[Dict]: trial 列表
    """
    trials = []
    
    # 查找所有 jsonl 文件
    files = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith(".jsonl"):
            files.append(os.path.join(data_dir, fname))
    
    if not files:
        raise FileNotFoundError(f"❌ 在 {data_dir} 中找不到 .jsonl 文件")
    
    files.sort()  # 确保顺序一致
    
    print(f"📂 找到 {len(files)} 个数据文件")
    
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    trial = json.loads(line)
                    trials.append(trial)
                    
                    if max_trials and len(trials) >= max_trials:
                        return trials
                except Exception as e:
                    print(f"⚠️ 解析失败: {e}")
                    continue
    
    return trials

def build_bm25_text(trial: Dict[str, Any]) -> str:
    """
    构建用于 BM25 检索的文档文本
    
    策略：
    - 侧重标题和疾病（高权重字段）
    - 入排标准只取前5条（减少噪音）
    - 适当包含干预措施
    """
    bg = trial.get("background", {}) or {}
    cr = trial.get("criteria", {}) or {}
    
    # 基础字段
    public_title = bg.get("public_title", "") or ""
    scientific_title = bg.get("scientific_title", "") or ""
    conditions = " ".join(bg.get("conditions", []) or [])
    interventions = " ".join(bg.get("interventions", []) or [])
    brief_summary = bg.get("brief_summary", "") or ""
    
    # 入排标准（截断以减少噪音）
    inclusion = cr.get("inclusion", []) or []
    exclusion = cr.get("exclusion", []) or []
    inc_text = " ".join(inclusion[:5])
    exc_text = " ".join(exclusion[:5])
    
    # 组合
    parts = [
        public_title,
        scientific_title,
        conditions,
        interventions,
        brief_summary,
        inc_text,
        exc_text,
    ]
    
    return normalize_text(" ".join(filter(None, parts)))

def get_bm25_instance(trials: List[Dict[str, Any]], cache_dir: str) -> BM25Okapi:
    """
    构建或加载 BM25 索引
    
    Args:
        trials: trial 列表
        cache_dir: 缓存目录
    
    Returns:
        BM25Okapi 实例
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"bm25_index_{len(trials)}.pkl")
    
    # 尝试加载缓存
    if os.path.exists(cache_file):
        print(f"📦 从缓存加载 BM25 索引: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    
    # 构建新索引
    print(f"🔨 构建 BM25 索引（共 {len(trials)} 个试验）...")
    
    corpus = []
    for i, trial in enumerate(trials):
        if (i + 1) % 5000 == 0:
            print(f"   进度: {i + 1}/{len(trials)}")
        
        text = build_bm25_text(trial)
        tokens = tokenize_zh(text)
        corpus.append(tokens)
    
    bm25 = BM25Okapi(corpus)
    
    # 保存缓存
    with open(cache_file, "wb") as f:
        pickle.dump(bm25, f)
    
    print(f"✅ BM25 索引已构建并缓存")
    return bm25

# ================= 核心改进：智能上下文剪裁 =================

def smart_truncate_list(items: List[str], head: int, tail: int) -> List[str]:
    """
    智能截断策略 - Head+Tail
    
    审稿人关注点：
    - Head（前N条）：通常是核心疾病定义、主要诊断标准
    - Tail（后N条）：通常是关键禁忌症（孕妇、过敏、严重并发症等）
    - Middle（中间）：通常是常规实验室指标，优先级较低
    
    Args:
        items: 完整列表
        head: 保留前几条
        tail: 保留后几条
    
    Returns:
        截断后的列表
    """
    if not items:
        return []
    
    # 如果总数不超过限制，直接返回
    if len(items) <= (head + tail):
        return items
    
    # Head + 省略标记 + Tail
    truncated = (
        items[:head] +
        [f"... [省略中间 {len(items) - head - tail} 条非关键条款] ..."] +
        items[-tail:]
    )
    
    return truncated

def format_trial_for_llm(trial: Dict[str, Any]) -> str:
    """
    将 trial 格式化为 LLM 友好的文本
    
    改进点：
    1. 使用 smart_truncate_list 而非粗暴截断
    2. 清晰的结构化格式
    3. 保留最重要的信息
    """
    trial_id = trial.get("trial_id", "Unknown")
    bg = trial.get("background", {}) or {}
    cr = trial.get("criteria", {}) or {}
    meta = trial.get("meta", {}) or {}
    
    # 基础信息
    title = bg.get("public_title") or bg.get("scientific_title") or "N/A"
    phase = meta.get("phase", "") or "N/A"
    conditions = bg.get("conditions", []) or []
    interventions = bg.get("interventions", []) or []
    
    # 智能截断入排标准
    inclusion_raw = cr.get("inclusion", []) or []
    exclusion_raw = cr.get("exclusion", []) or []
    
    inclusion = smart_truncate_list(inclusion_raw, CRITERIA_HEAD_KEEP, CRITERIA_TAIL_KEEP)
    exclusion = smart_truncate_list(exclusion_raw, CRITERIA_HEAD_KEEP, CRITERIA_TAIL_KEEP)
    
    # 格式化
    def format_list(lst):
        return "\n".join([f"  - {item}" for item in lst])
    
    context = f"""Trial ID: {trial_id}
Title: {title}
Phase: {phase}
Conditions: {', '.join(conditions) if conditions else 'N/A'}
Interventions: {', '.join(interventions[:5]) if interventions else 'N/A'}

[Inclusion Criteria]
{format_list(inclusion) if inclusion else '  - N/A'}

[Exclusion Criteria]
{format_list(exclusion) if exclusion else '  - N/A'}
"""
    
    return context

# ================= 核心改进：鲁棒的 LLM Prompt =================

LLM_SYSTEM_PROMPT = """You are a Clinical Trial Matching Assistant with medical expertise.

**Your Task**: Compare the Patient Profile with the Clinical Trial Protocol and make a matching decision.

**Decision Logic**:
1. **Included**: 
   - Patient likely MEETS the main inclusion criteria based on available information
   - AND has NO obvious exclusion factors
   - Make reasonable clinical judgments when minor details are missing
   - If disease type, stage, and key biomarkers match, lean towards Included

2. **Excluded**: 
   - Patient clearly VIOLATES at least one major exclusion criterion
   - OR clearly FAILS to meet critical inclusion criteria (e.g., wrong disease, wrong stage)
   - Disease type mismatch or obvious contraindications

3. **Insufficient**: 
   - Patient profile LACKS CRITICAL information that makes ANY judgment impossible
   - Examples: Disease type unknown, stage completely missing for stage-specific trials, no biomarker data when trial specifically requires it
   - ONLY use when truly cannot make a reasonable medical judgment
   - NOT for minor missing details (e.g., exact body temperature, specific lab test dates)

**Output Requirements**:
You MUST output a valid JSON object with these exact fields:
{
  "label": "Included" | "Excluded" | "Insufficient",
  "score": <number 0-100>,
  "reason": "<Brief summary in Chinese, 1-2 sentences>",
  "evidence": "<Quote the SPECIFIC criteria text that led to this decision>"
}

**Score Guidelines**:
- Excluded: 0-40 (clear mismatch or contraindication)
- Insufficient: 40-60 (truly missing critical core information)
- Included: 60-100 (reasonable match, higher = stronger match)

**Critical Rules**:
- ALWAYS quote specific criteria in "evidence"
- Make reasonable clinical judgments - missing minor details should NOT lead to Insufficient
- Choose "Insufficient" ONLY when core diagnostic information is missing
- Never fabricate information not in the patient profile
- Focus on practical trial matching, not information perfection
- When in doubt between Included and Insufficient, if core info (disease, stage, key biomarkers) is present, lean towards Included
"""

def llm_rerank(
    client: OpenAI,
    model: str,
    patient_text: str,
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    使用 LLM 对候选试验进行重排序
    
    改进点：
    1. 强制 JSON 输出模式
    2. 要求 evidence 引用
    3. 详细的进度显示
    4. 更好的错误处理
    
    Args:
        client: OpenAI client
        model: 模型名称
        patient_text: 患者描述
        candidates: 候选试验列表
    
    Returns:
        排序后的结果列表
    """
    results = []
    total = len(candidates)
    
    print(f"\n🤖 开始 AI 分析（共 {total} 个候选，模型: {model}）...")
    print("=" * 70)
    
    for i, trial in enumerate(candidates, 1):
        trial_context = format_trial_for_llm(trial)
        
        user_message = f"""[Patient Profile]
{patient_text}

[Clinical Trial Protocol]
{trial_context}

Please evaluate the match and output JSON."""
        
        try:
            start_time = time.time()
            
            # 强制 JSON 输出模式（DeepSeek 支持）
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"},  # 强制 JSON
                temperature=0.0,  # 确保一致性
            )
            
            latency = time.time() - start_time
            
            # 解析响应
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            # 提取字段
            label = parsed.get("label", "Insufficient")
            score = float(parsed.get("score", 0))
            reason = parsed.get("reason", "")
            evidence = parsed.get("evidence", "")
            
            # 构建结果
            result = {
                "trial_id": trial.get("trial_id", ""),
                "title": trial.get("background", {}).get("public_title", ""),
                "label": label,
                "score": score,
                "reason": reason,
                "evidence": evidence,
                "latency": latency,
            }
            
            results.append(result)
            
            # 进度显示
            status_icon = {
                "Included": "✅",
                "Excluded": "❌",
                "Insufficient": "⚠️"
            }.get(label, "❓")
            
            print(f"[{i:2d}/{total}] {status_icon} {trial['trial_id']:12s} | "
                  f"{label:12s} | Score: {score:5.1f} | {latency:.1f}s")
            
        except json.JSONDecodeError as e:
            print(f"[{i:2d}/{total}] ❌ JSON 解析失败: {trial['trial_id']}")
            results.append({
                "trial_id": trial.get("trial_id", ""),
                "label": "Error",
                "score": 0,
                "reason": f"JSON解析错误: {str(e)}",
                "evidence": "",
                "latency": 0,
            })
        
        except Exception as e:
            print(f"[{i:2d}/{total}] ❌ 处理失败: {trial['trial_id']} - {str(e)}")
            results.append({
                "trial_id": trial.get("trial_id", ""),
                "label": "Error",
                "score": 0,
                "reason": f"处理错误: {str(e)}",
                "evidence": "",
                "latency": 0,
            })
    
    print("=" * 70)
    
    # 按分数降序排序
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results

# ================= BM25 检索 =================

def bm25_retrieve(
    bm25: BM25Okapi,
    trials: List[Dict[str, Any]],
    query: str,
    topk: int
) -> List[Dict[str, Any]]:
    """
    使用 BM25 检索候选试验
    
    Args:
        bm25: BM25 索引
        trials: 全部试验
        query: 查询文本
        topk: 返回数量
    
    Returns:
        候选试验列表
    """
    query_tokens = tokenize_zh(query)
    
    if not query_tokens:
        print("⚠️ 查询分词结果为空，请检查输入")
        return []
    
    # 计算分数
    scores = bm25.get_scores(query_tokens)
    
    # 排序并取 topk
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
    
    candidates = [trials[i] for i in indices]
    
    return candidates

# ================= 主程序 =================

def main():
    parser = argparse.ArgumentParser(
        description="TrialGPT-China V2 - SIGIR Rebuttal Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python TrialGPT_China_V2.py --data_dir "D:\\实习\\TrialGPT-China\\cleaned_out_V4\\clean_parts" --topk 50 --topn 10
  
  python TrialGPT_China_V2.py --data_dir "D:\\实习\\TrialGPT-China\\cleaned_out_V4\\clean_parts" --query "患者描述..." --topk 30

环境变量:
  DEEPSEEK_API_KEY - DeepSeek API 密钥
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="清洗后数据目录（包含 *.jsonl 文件）"
    )
    
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="患者描述文本（不提供则交互式输入）"
    )
    
    parser.add_argument(
        "--topk",
        type=int,
        default=50,
        help="BM25 召回数量（default: 50）"
    )
    
    parser.add_argument(
        "--topn",
        type=int,
        default=10,
        help="最终输出数量（default: 10）"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-chat",
        help="DeepSeek 模型名称（default: deepseek-chat）"
    )
    
    parser.add_argument(
        "--max_trials",
        type=int,
        default=0,
        help="最多加载试验数量（0 = 全部，用于调试）"
    )
    
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="保存结果到 JSON 文件（可选）"
    )
    
    args = parser.parse_args()
    
    # ===== 1. 加载数据 =====
    print("\n" + "=" * 70)
    print("TrialGPT-China V2 - Clinical Trial Matching System")
    print("=" * 70)
    print(f"\n📚 [1/4] 加载数据...")
    
    trials = load_trials(args.data_dir, max_trials=args.max_trials)
    print(f"✅ 已加载 {len(trials)} 个试验")
    
    # ===== 2. 构建/加载 BM25 索引 =====
    print(f"\n🔍 [2/4] 准备检索索引...")
    bm25 = get_bm25_instance(trials, cache_dir="./cache")
    
    # ===== 3. 获取患者描述 =====
    print(f"\n👤 [3/4] 获取患者信息...")
    
    patient_text = args.query
    
    if not patient_text:
        print("请输入患者描述（输入 END 单独一行结束）：")
        print("-" * 70)
        lines = []
        while True:
            try:
                line = input()
                if line.strip().upper() == "END":
                    break
                lines.append(line)
            except EOFError:
                break
        
        patient_text = normalize_text("\n".join(lines))
    
    if not patient_text:
        print("❌ 患者描述为空，退出")
        return
    
    print(f"✅ 患者描述长度: {len(patient_text)} 字符")
    print(f"   预览: {patient_text[:100]}...")
    
    # ===== 4. BM25 检索 + LLM 重排序 =====
    print(f"\n🔍 [4/4] 检索与匹配...")
    
    # BM25 召回
    print(f"\n📊 BM25 检索中（topk={args.topk}）...")
    candidates = bm25_retrieve(bm25, trials, patient_text, args.topk)
    print(f"✅ BM25 召回 {len(candidates)} 个候选")
    
    # 检查 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print("\n❌ 错误: 未设置环境变量 DEEPSEEK_API_KEY")
        print("请在 PowerShell 执行: setx DEEPSEEK_API_KEY \"sk-xxxx\"")
        print("然后重开终端/VSCode")
        return
    
    # 初始化 OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )
    
    # LLM 重排序
    ranked_results = llm_rerank(client, args.model, patient_text, candidates)
    
    # ===== 5. 输出结果 =====
    print("\n" + "=" * 70)
    print(f"📋 最终推荐结果（Top {args.topn}）")
    print("=" * 70)
    
    for i, result in enumerate(ranked_results[:args.topn], 1):
        label = result["label"]
        score = result["score"]
        trial_id = result["trial_id"]
        title = result.get("title", "")
        reason = result.get("reason", "")
        evidence = result.get("evidence", "")
        
        # 状态图标
        status_icon = {
            "Included": "✅",
            "Excluded": "❌",
            "Insufficient": "⚠️",
            "Error": "❌"
        }.get(label, "❓")
        
        print(f"\n{i}. {status_icon} [{label}] Score: {score:.1f} | ID: {trial_id}")
        
        if title:
            print(f"   标题: {title[:80]}{'...' if len(title) > 80 else ''}")
        
        print(f"   理由: {reason}")
        
        if evidence:
            evidence_preview = evidence[:150] + "..." if len(evidence) > 150 else evidence
            print(f"   引用: \"{evidence_preview}\"")
        
        print("-" * 70)
    
    # ===== 6. 保存结果（可选）=====
    if args.save_json:
        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "topk": args.topk,
                "topn": args.topn,
                "total_trials": len(trials),
            },
            "patient_text": patient_text,
            "top_results": ranked_results[:args.topn],
            "all_results": ranked_results,
        }
        
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 结果已保存到: {args.save_json}")
    
    # ===== 统计信息 =====
    print("\n" + "=" * 70)
    print("📊 统计信息")
    print("=" * 70)
    
    label_counts = {}
    for r in ranked_results:
        label = r["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    for label, count in sorted(label_counts.items()):
        print(f"  {label:12s}: {count:3d}")
    
    avg_latency = sum(r.get("latency", 0) for r in ranked_results) / len(ranked_results) if ranked_results else 0
    print(f"\n  平均响应时间: {avg_latency:.2f}s")
    print(f"  总处理时间: {sum(r.get('latency', 0) for r in ranked_results):.1f}s")
    
    print("\n✅ 完成！")


if __name__ == "__main__":
    main()
