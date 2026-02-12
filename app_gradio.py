# -*- coding: utf-8 -*-
"""
TrialRAG Web Interface (Gradio)

一个美观的 Web 界面，用于临床试验匹配

安装依赖：
pip install gradio --break-system-packages

运行：
python app_gradio.py

然后在浏览器打开显示的 URL（通常是 http://127.0.0.1:7860）
"""

import os
import json
import gradio as gr
from datetime import datetime
from typing import List, Dict, Any

# 导入主程序的核心功能
import sys
sys.path.append(os.path.dirname(__file__))

from TrialRAG_V2 import (
    load_trials,
    get_bm25_instance,
    bm25_retrieve,
    llm_rerank,
    normalize_text
)
from openai import OpenAI


# ================= 全局配置 =================

# 默认配置
DEFAULT_DATA_DIR = r"D:\实习\TrialGPT-China\cleaned_out_V4\clean_parts"
DEFAULT_MODEL = "deepseek-chat"

# 全局变量（缓存）
TRIALS_CACHE = None
BM25_CACHE = None
DATA_DIR_CACHE = None


# ================= 核心功能函数 =================

def initialize_system(data_dir: str, progress=gr.Progress()):
    """初始化系统：加载数据和构建索引"""
    global TRIALS_CACHE, BM25_CACHE, DATA_DIR_CACHE
    
    # 检查是否需要重新加载
    if DATA_DIR_CACHE == data_dir and TRIALS_CACHE is not None:
        return True, f"✅ 系统已就绪（{len(TRIALS_CACHE)} 个试验）"
    
    try:
        # 加载试验数据
        progress(0.3, desc="加载试验数据...")
        trials = load_trials(data_dir, max_trials=0)
        
        if not trials:
            return False, f"❌ 错误：在 {data_dir} 中未找到试验数据"
        
        # 构建 BM25 索引
        progress(0.6, desc="构建检索索引...")
        bm25 = get_bm25_instance(trials, cache_dir="./cache")
        
        # 更新缓存
        TRIALS_CACHE = trials
        BM25_CACHE = bm25
        DATA_DIR_CACHE = data_dir
        
        progress(1.0, desc="初始化完成！")
        return True, f"✅ 系统就绪：已加载 {len(trials)} 个临床试验"
        
    except Exception as e:
        return False, f"❌ 初始化失败：{str(e)}"


def search_trials(
    patient_text: str,
    topk: int,
    topn: int,
    model: str,
    api_key: str,
    progress=gr.Progress()
):
    """执行临床试验匹配"""
    
    # 检查输入
    if not patient_text or not patient_text.strip():
        return "❌ 请输入患者描述", "", ""
    
    if not api_key or not api_key.strip():
        return "❌ 请设置 DeepSeek API Key", "", ""
    
    # 检查系统是否已初始化
    if TRIALS_CACHE is None or BM25_CACHE is None:
        return "❌ 请先点击「初始化系统」按钮", "", ""
    
    try:
        patient_text = normalize_text(patient_text)
        
        # Step 1: BM25 检索
        progress(0.2, desc=f"BM25 检索中（topk={topk}）...")
        candidates = bm25_retrieve(BM25_CACHE, TRIALS_CACHE, patient_text, topk)
        
        if not candidates:
            return "⚠️ 未找到相关试验，请检查患者描述", "", ""
        
        # Step 2: LLM 重排序
        progress(0.4, desc=f"DeepSeek 重排序中（{len(candidates)} 个候选）...")
        
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # 进度更新回调
        def update_progress(current, total):
            progress((0.4 + 0.5 * current / total), 
                    desc=f"分析中 [{current}/{total}]...")
        
        ranked_results = llm_rerank_with_progress(
            client, model, patient_text, candidates, update_progress
        )
        
        progress(0.95, desc="生成结果...")
        
        # Step 3: 格式化输出
        results_html = format_results_html(ranked_results[:topn])
        results_json = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "patient_text": patient_text,
            "model": model,
            "topk": topk,
            "topn": topn,
            "results": ranked_results[:topn]
        }, ensure_ascii=False, indent=2)
        
        # 统计信息
        stats = generate_statistics(ranked_results)
        
        progress(1.0, desc="完成！")
        
        return results_html, results_json, stats
        
    except Exception as e:
        return f"❌ 匹配失败：{str(e)}", "", ""


def llm_rerank_with_progress(client, model, patient_text, candidates, progress_callback):
    """带进度回调的 LLM 重排序"""
    from TrialRAG_V2 import format_trial_for_llm, LLM_SYSTEM_PROMPT
    
    results = []
    total = len(candidates)
    
    for i, trial in enumerate(candidates, 1):
        try:
            trial_context = format_trial_for_llm(trial)
            
            user_prompt = f"""Patient Description:
{patient_text}

Clinical Trial:
{trial_context}

Provide your assessment in JSON format ONLY."""
            
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            parsed = json.loads(content)
            
            results.append({
                "trial_id": trial.get("trial_id", ""),
                "title": trial.get("background", {}).get("public_title", ""),
                "label": parsed.get("label", "Insufficient"),
                "score": float(parsed.get("score", 0)),
                "reason": parsed.get("reason", ""),
                "evidence": parsed.get("evidence", ""),
            })
            
            # 更新进度
            if progress_callback:
                progress_callback(i, total)
                
        except Exception as e:
            results.append({
                "trial_id": trial.get("trial_id", ""),
                "title": "",
                "label": "Error",
                "score": 0,
                "reason": f"处理错误: {str(e)}",
                "evidence": "",
            })
    
    # 按分数降序排序
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def format_results_html(results: List[Dict[str, Any]]) -> str:
    """格式化结果为 HTML"""
    
    html = """
    <style>
        .result-container { font-family: Arial, sans-serif; }
        .result-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background: white;
        }
        .result-card.included { border-left: 4px solid #10b981; }
        .result-card.excluded { border-left: 4px solid #ef4444; }
        .result-card.insufficient { border-left: 4px solid #f59e0b; }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .result-label {
            padding: 4px 12px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }
        .label-included { background: #d1fae5; color: #065f46; }
        .label-excluded { background: #fee2e2; color: #991b1b; }
        .label-insufficient { background: #fef3c7; color: #92400e; }
        .result-score {
            font-size: 24px;
            font-weight: bold;
            color: #1f2937;
        }
        .result-title { 
            font-size: 16px;
            font-weight: bold;
            color: #1f2937;
            margin-bottom: 8px;
        }
        .result-reason {
            color: #4b5563;
            margin-bottom: 8px;
            line-height: 1.6;
        }
        .result-evidence {
            background: #f9fafb;
            border-left: 3px solid #e5e7eb;
            padding: 10px;
            margin-top: 8px;
            font-size: 13px;
            color: #6b7280;
            font-style: italic;
        }
        .result-id {
            color: #9ca3af;
            font-size: 12px;
        }
    </style>
    <div class="result-container">
    """
    
    for i, result in enumerate(results, 1):
        label = result["label"]
        score = result["score"]
        
        # 确定样式类
        card_class = label.lower() if label.lower() in ["included", "excluded", "insufficient"] else "insufficient"
        label_class = f"label-{card_class}"
        
        # 标签图标
        icon = {"included": "✅", "excluded": "❌", "insufficient": "⚠️"}.get(card_class, "❓")
        
        html += f"""
        <div class="result-card {card_class}">
            <div class="result-header">
                <span class="result-label {label_class}">{icon} {label}</span>
                <span class="result-score">{score:.1f}</span>
            </div>
            <div class="result-title">{i}. {result.get('title', '未知试验')}</div>
            <div class="result-id">Trial ID: {result['trial_id']}</div>
            <div class="result-reason"><strong>理由：</strong>{result['reason']}</div>
            <div class="result-evidence"><strong>引用：</strong>"{result['evidence'][:200]}{'...' if len(result['evidence']) > 200 else ''}"</div>
        </div>
        """
    
    html += "</div>"
    return html


def generate_statistics(results: List[Dict[str, Any]]) -> str:
    """生成统计信息"""
    label_counts = {}
    for r in results:
        label = r["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    stats = f"""
📊 **匹配统计**

- 总候选数：{len(results)}
- ✅ Included：{label_counts.get('Included', 0)} 个
- ❌ Excluded：{label_counts.get('Excluded', 0)} 个
- ⚠️ Insufficient：{label_counts.get('Insufficient', 0)} 个
- ❓ Error：{label_counts.get('Error', 0)} 个

---

**平均分数：** {sum(r['score'] for r in results) / len(results):.2f}

**Top 5 分数：** {', '.join([f"{r['score']:.1f}" for r in results[:5]])}
    """
    
    return stats


# ================= Gradio 界面 =================

def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(
        title="TrialRAG - 临床试验智能匹配系统",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🏥 TrialRAG - 临床试验智能匹配系统
        
        基于 BM25 检索 + DeepSeek 大模型重排序的临床试验匹配系统
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 左侧：输入区域
                gr.Markdown("## 📝 患者信息")
                
                patient_input = gr.Textbox(
                    label="患者描述",
                    placeholder="""请输入患者详细信息，包括：
- 基本信息：年龄、性别
- 诊断：疾病类型、分期
- 分子标志物：EGFR、ALK、HER2 等
- 既往治疗史
- 体能状态：ECOG 评分
- 实验室检查：血常规、肝肾功能
- 合并症

示例：
患者男性62岁，诊断为晚期非小细胞肺癌，临床分期IV期。病理类型为肺腺癌。EGFR基因19号外显子缺失突变阳性，ALK融合基因阴性。患者为初诊，尚未接受任何系统性抗肿瘤治疗。ECOG体能状态评分1分，体重65公斤。血常规、肝肾功能正常。无其他严重疾病。""",
                    lines=15
                )
                
                gr.Markdown("## ⚙️ 系统配置")
                
                
                data_dir_input = gr.Textbox(
                    label="数据目录路径",
                    value=DEFAULT_DATA_DIR,
                    placeholder=r"D:\实习\TrialGPT-China\cleaned_out_V4\clean_parts"
                )
                init_btn = gr.Button("🔄 初始化系统", variant="secondary")
                init_status = gr.Markdown("💡 请先初始化系统")
                
                with gr.Accordion("检索参数", open=True):
                    topk_slider = gr.Slider(
                        minimum=10,
                        maximum=100,
                        value=30,
                        step=10,
                        label="BM25 召回数量 (topk)",
                        info="初步召回的候选试验数量"
                    )
                    
                    topn_slider = gr.Slider(
                        minimum=5,
                        maximum=20,
                        value=10,
                        step=5,
                        label="最终推荐数量 (topn)",
                        info="经过 LLM 重排序后展示的数量"
                    )
                
                with gr.Accordion("模型设置", open=False):
                    model_input = gr.Textbox(
                        label="DeepSeek 模型",
                        value=DEFAULT_MODEL,
                        placeholder="deepseek-chat"
                    )
                    
                    api_key_input = gr.Textbox(
                        label="DeepSeek API Key",
                        type="password",
                        placeholder="sk-xxxx",
                        value=os.getenv("DEEPSEEK_API_KEY", "")
                    )
                
                search_btn = gr.Button("🔍 开始匹配", variant="primary", size="lg")
            
            with gr.Column(scale=2):
                # 右侧：结果区域
                gr.Markdown("## 📊 匹配结果")
                
                with gr.Tabs():
                    with gr.Tab("可视化结果"):
                        results_html = gr.HTML(label="匹配结果")
                    
                    with gr.Tab("JSON 数据"):
                        results_json = gr.Code(
                            label="JSON 结果",
                            language="json",
                            lines=20
                        )
                    
                    with gr.Tab("统计信息"):
                        stats_output = gr.Markdown(label="统计")
        
        # 事件绑定
        init_btn.click(
            fn=initialize_system,
            inputs=[data_dir_input],
            outputs=[gr.State(), init_status]
        )
        
        search_btn.click(
            fn=search_trials,
            inputs=[
                patient_input,
                topk_slider,
                topn_slider,
                model_input,
                api_key_input
            ],
            outputs=[results_html, results_json, stats_output]
        )
        
        # 示例
        gr.Markdown("---")
        gr.Markdown("### 💡 快速示例")
        
        gr.Examples(
            examples=[
                ["患者男性62岁，诊断为晚期非小细胞肺癌，临床分期IV期。病理类型为肺腺癌。EGFR基因19号外显子缺失突变阳性，ALK融合基因阴性，PD-L1表达TPS 5%。患者为初诊，尚未接受任何系统性抗肿瘤治疗，未接受过放疗或手术。ECOG体能状态评分1分，体重65公斤，身高172厘米。血常规白细胞6.2，血红蛋白125，血小板180。肝肾功能正常。合并高血压药物控制良好。无其他严重疾病。患者愿意配合临床试验。"],
                ["患者女性58岁，HER2阳性乳腺癌，已完成新辅助化疗，计划进行手术治疗。心功能正常，LVEF 60%。无其他重大合并症。"],
                ["患者男性55岁，胃腺癌，临床分期III期，HER2阴性，计划接受围手术期化疗。ECOG评分1分。"],
            ],
            inputs=[patient_input],
        )
    
    return demo


# ================= 主程序 =================

if __name__ == "__main__":
    print("=" * 70)
    print("TrialRAG Web Interface")
    print("=" * 70)
    print("\n启动 Gradio 服务器...")
    print("启动后请在浏览器中打开显示的 URL\n")
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=False,  # 设为 True 可生成公网链接
        show_error=True
    )
