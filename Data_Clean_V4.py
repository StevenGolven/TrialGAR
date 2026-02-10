# -*- coding: utf-8 -*-


import os
import re
import json
import random
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from xml.etree import ElementTree as ET

# ====================== 配置 ======================

ROOT_DIR = r"D:\实习\all-detail"
OUT_DIR = r"D:\实习\TrialGPT-China\cleaned_out_V4"

SHARD_SIZE = 5000
SAMPLE_N = 50

# Word 2003 XML 命名空间
NS = {
    'w': 'http://schemas.microsoft.com/office/word/2003/wordml',
    'o': 'urn:schemas-microsoft-com:office:office'
}

# ====================== Format-Aware DOM Parser ======================

class ParagraphMetadata:
    """段落元数据：文本 + 格式特征"""
    def __init__(self, text: str, is_bold: bool = False, font_size: int = 0):
        self.text = text.strip()
        self.is_bold = is_bold
        self.font_size = font_size
        self.is_likely_header = is_bold and len(text.strip()) < 30
    
    def __repr__(self):
        bold_mark = "**" if self.is_bold else ""
        return f"{bold_mark}{self.text}{bold_mark}"


class FormatAwareDocParser:
    """
    格式感知的Word XML解析器
    核心改进：提取加粗、字号等格式特征，用作语义代理
    """
    
    def __init__(self, file_path: str):
        self.path = file_path
        self.raw_content = ""
        self.root = None
        self.paragraphs = []  # List[ParagraphMetadata]
        self.ns = NS
        
    def parse_dom(self) -> bool:
        """
        DOM树解析（优先策略）
        提取段落文本 + 格式特征（加粗、字号）
        """
        try:
            with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            self.raw_content = content
            
            # 预处理CDATA
            content = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', content, flags=re.DOTALL)
            
            # 解析XML树
            self.root = ET.fromstring(content)
            
            # 遍历所有段落 (w:p)
            for p_elem in self.root.findall('.//w:p', self.ns):
                p_text = ""
                is_bold = False
                font_size = 0
                
                # 遍历段落内的所有run (w:r)
                for r_elem in p_elem.findall('.//w:r', self.ns):
                    # 提取格式属性 (w:rPr - Run Properties)
                    rPr = r_elem.find('w:rPr', self.ns)
                    if rPr is not None:
                        # 检查加粗 (w:b)
                        if rPr.find('w:b', self.ns) is not None:
                            is_bold = True
                        
                        # 检查字号 (w:sz)
                        sz_elem = rPr.find('w:sz', self.ns)
                        if sz_elem is not None:
                            try:
                                # Word XML字号单位是半点（1pt = 2单位）
                                font_size = max(font_size, int(sz_elem.get('{' + self.ns['w'] + '}val', 0)) // 2)
                            except:
                                pass
                    
                    # 提取文本 (w:t)
                    t_elem = r_elem.find('w:t', self.ns)
                    if t_elem is not None and t_elem.text:
                        p_text += t_elem.text
                
                full_text = p_text.strip()
                if full_text:
                    # 创建带格式的段落对象
                    para = ParagraphMetadata(full_text, is_bold, font_size)
                    self.paragraphs.append(para)
            
            return len(self.paragraphs) > 0
            
        except ET.ParseError:
            return False
        except Exception:
            return False
    
    def parse_regex_fallback(self) -> bool:
        """
        正则回退策略（当DOM解析失败时）
        论文卖点：Robustness through hybrid approach
        """
        if not self.raw_content:
            try:
                with open(self.path, 'r', encoding='utf-8', errors='ignore') as f:
                    self.raw_content = f.read()
            except:
                return False
        
        # 提取所有<w:t>节点
        text_nodes = re.findall(r'<w:t[^>]*>(.*?)</w:t>', self.raw_content, re.DOTALL)
        
        texts = []
        for node in text_nodes:
            # 处理CDATA
            if '<![CDATA[' in node:
                cdata = re.findall(r'<!\[CDATA\[(.*?)\]\]>', node, re.DOTALL)
                texts.extend(cdata)
            else:
                clean = re.sub(r'<[^>]+>', '', node)
                if clean.strip():
                    texts.append(clean.strip())
        
        if len(texts) < 10:
            # 最后的回退：直接提取CDATA
            cdata_blocks = re.findall(r'<!\[CDATA\[(.*?)\]\]>', self.raw_content, re.DOTALL)
            texts = cdata_blocks
        
        # 转换为ParagraphMetadata（但没有格式信息）
        for text in texts:
            if text.strip():
                self.paragraphs.append(ParagraphMetadata(text.strip()))
        
        return len(self.paragraphs) > 0
    
    def parse(self) -> bool:
        """
        统一解析入口
        先尝试DOM，失败则回退到正则
        """
        # 策略1: DOM解析（可以提取格式）
        if self.parse_dom():
            return True
        
        # 策略2: 正则回退（无格式信息，但更鲁棒）
        return self.parse_regex_fallback()
    
    def get_paragraphs(self) -> List[ParagraphMetadata]:
        """获取所有段落（带格式）"""
        return self.paragraphs
    
    def get_plain_text(self) -> str:
        """获取纯文本（向后兼容）"""
        return ' '.join([p.text for p in self.paragraphs])


# ====================== Format-Aware Field Extractor ======================

class StateBasedExtractor:
    """
    基于状态机的字段提取器
    核心改进：利用格式特征（加粗）辅助section切换
    """
    
    def __init__(self, parser: FormatAwareDocParser):
        self.parser = parser
        self.paragraphs = parser.get_paragraphs()
        self.full_text = parser.get_plain_text()
    
    def extract_trial_id(self) -> str:
        """
        提取Trial ID（三重策略）
        """
        # 策略1: 从文件名
        filename = os.path.basename(self.parser.path)
        m = re.search(r'(CTR\d{8})', filename)
        if m:
            return m.group(1)
        
        # 策略2: 从文本
        m = re.search(r'(CTR\d{8})', self.full_text)
        if m:
            return m.group(1)
        
        # 策略3: 从"登记号"附近
        for i, para in enumerate(self.paragraphs):
            if '登记号' in para.text or '注册号' in para.text:
                # 查看后续几个段落
                for j in range(i, min(i+5, len(self.paragraphs))):
                    m = re.search(r'(CTR\d{8})', self.paragraphs[j].text)
                    if m:
                        return m.group(1)
        
        return ""
    
    def _find_field_with_format(self, keywords: List[str]) -> str:
        """
        利用格式特征辅助字段查找
        关键改进：如果看到加粗的短文本包含关键词，优先认为它是标题
        """
        for i, para in enumerate(self.paragraphs):
            text = para.text
            
            # 检查关键词
            for keyword in keywords:
                if keyword in text:
                    # 如果是加粗的短文本，很可能是标题，取后续内容
                    if para.is_likely_header:
                        # 收集后续段落直到下一个标题
                        content = []
                        for j in range(i+1, min(i+15, len(self.paragraphs))):
                            next_para = self.paragraphs[j]
                            # 遇到下一个标题就停止
                            if next_para.is_likely_header:
                                break
                            content.append(next_para.text)
                        
                        result = ' '.join(content).strip()
                        if result:
                            return result
                    else:
                        # 不是标题，可能关键词和内容在同一段
                        # 去掉关键词本身
                        result = text.replace(keyword, '', 1).strip()
                        if result:
                            return result
        
        return ""
    
    def extract_titles(self) -> Tuple[str, str]:
        """提取试验题目"""
        public = self._find_field_with_format(['试验通俗题目', '公示标题', '通俗题目'])
        scientific = self._find_field_with_format(['试验专业题目', '科学标题', '专业题目'])
        return public, scientific
    
    def extract_conditions(self) -> List[str]:
        """提取适应症"""
        text = self._find_field_with_format(['适应症', '疾病', '病症'])
        if not text:
            return []
        
        parts = re.split(r'[;；,，/、\n]+', text)
        conditions = []
        for p in parts:
            p = p.strip()
            if 1 < len(p) < 50:
                conditions.append(p)
        
        return conditions[:10]
    
    def extract_phase(self) -> str:
        """提取试验分期"""
        text = self._find_field_with_format(['试验分期', '研究阶段', '临床分期'])
        if not text:
            return ""
        
        phase_map = {
            'Ⅰ': 'I期', 'Ⅱ': 'II期', 'Ⅲ': 'III期', 'Ⅳ': 'IV期',
            'I期': 'I期', 'II期': 'II期', 'III期': 'III期', 'IV期': 'IV期',
            'I/II': 'I/II期', 'II/III': 'II/III期',
        }
        
        for key, val in phase_map.items():
            if key in text:
                return val
        
        return text[:20]
    
    def extract_interventions(self) -> List[str]:
        """提取干预措施"""
        text = self._find_field_with_format(['药物名称', '试验药物', '干预措施', '药物信息'])
        if not text:
            return []
        
        # 去噪音
        text = re.sub(r'(联系人|电话|邮箱|地址|单位).{0,30}', ' ', text)
        
        # 提取药物名
        tokens = re.findall(r'[A-Za-z0-9\-]{3,}|[\u4e00-\u9fff]{2,10}', text)
        
        bad_words = {'对照', '常规', '标准', '治疗', '方案', '药物', '企业', '公示'}
        
        drugs = []
        for token in tokens:
            token = token.strip()
            if len(token) < 2 or token in bad_words:
                continue
            if token not in drugs:
                drugs.append(token)
        
        return drugs[:12]
    
    def extract_brief_summary(self) -> str:
        """提取试验目的"""
        return self._find_field_with_format(['试验目的', '研究目的', '试验简介'])
    
    def extract_criteria_format_aware(self, criteria_type: str) -> List[str]:
        """
        格式感知的入排标准提取
        
        核心改进：
        1. 利用加粗判断section标题
        2. 状态机驱动，避免误判
        3. 结合编号特征
        
        criteria_type: 'inclusion' or 'exclusion'
        """
        if criteria_type == 'inclusion':
            start_keywords = ['入选标准', '纳入标准', '入组标准']
            end_keywords = ['排除标准', '剔除标准', '主要结局', '次要结局']
        else:
            start_keywords = ['排除标准', '剔除标准']
            end_keywords = ['主要结局', '次要结局', '研究者', '申办者', '伦理']
        
        criteria_items = []
        current_state = 'SEEKING'  # 状态：SEEKING -> IN_SECTION -> ENDED
        
        for i, para in enumerate(self.paragraphs):
            text = para.text
            clean_text = text.replace(' ', '')
            
            # === 状态1: 寻找section起始 ===
            if current_state == 'SEEKING':
                # 检查是否匹配起始关键词
                for keyword in start_keywords:
                    if keyword in clean_text:
                        # ✅ 格式感知判断：加粗 OR 短文本 OR 编号开头
                        is_section_header = (
                            para.is_likely_header or 
                            len(clean_text) < 15 or
                            re.match(r'^[\d一二三四]+[\.、]', clean_text)
                        )
                        
                        if is_section_header:
                            current_state = 'IN_SECTION'
                            break  # 找到了，跳过标题本身
            
            # === 状态2: 收集section内容 ===
            elif current_state == 'IN_SECTION':
                # 检查是否遇到结束关键词
                for end_kw in end_keywords:
                    if end_kw in clean_text:
                        # ✅ 格式感知：如果是加粗标题，确认结束
                        if para.is_likely_header or len(clean_text) < 15:
                            current_state = 'ENDED'
                            break
                
                if current_state == 'ENDED':
                    break
                
                # 收集内容（过滤明显的噪音）
                if len(text) >= 5:
                    if not any(noise in text for noise in ['联系人', '电话', '邮箱', '医院名称']):
                        criteria_items.append(text)
        
        # 智能拆分：如果提取的条目太少，可能是粘在一起的
        if len(criteria_items) == 0:
            # 回退：用原来的方法
            return self._extract_criteria_fallback(criteria_type)
        elif len(criteria_items) == 1 and len(criteria_items[0]) > 500:
            # 单条太长，需要拆分
            return self._split_long_criteria(criteria_items[0])
        
        return criteria_items[:30]
    
    def _extract_criteria_fallback(self, criteria_type: str) -> List[str]:
        """回退方法：基于完整文本的区间提取"""
        if criteria_type == 'inclusion':
            keywords = ['入选标准', '纳入标准', '入组标准']
            end_kws = ['排除标准', '剔除标准']
        else:
            keywords = ['排除标准', '剔除标准']
            end_kws = ['主要结局', '次要结局', '研究者']
        
        criteria_text = ""
        for kw in keywords:
            if kw in self.full_text:
                start = self.full_text.index(kw) + len(kw)
                end = len(self.full_text)
                
                for end_kw in end_kws:
                    idx = self.full_text.find(end_kw, start)
                    if idx != -1 and idx < end:
                        end = idx
                
                criteria_text = self.full_text[start:end]
                break
        
        if not criteria_text:
            return []
        
        return self._split_long_criteria(criteria_text)
    
    def _split_long_criteria(self, text: str) -> List[str]:
        """智能拆分长文本"""
        items = []
        
        # 统一替换编号
        text_marked = text
        text_marked = re.sub(r'(\d+)\s*[.、\.。)）]\s*', r'|||ITEM\1||| ', text_marked)
        text_marked = re.sub(r'[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳]', '|||ITEM|||', text_marked)
        text_marked = re.sub(r'[（\(]\s*(\d+)\s*[）\)]', r'|||ITEM\1|||', text_marked)
        
        parts = text_marked.split('|||ITEM')
        
        for part in parts:
            part = part.replace('|||', '').strip()
            part = re.sub(r'^\d+\s*', '', part)
            part = part.strip('；;，,。. \t\n')
            
            if len(part) < 5:
                continue
            
            # 二次拆分：如果仍然很长
            if len(part) > 500:
                sub_parts = part.split('；')
                items.extend([p.strip() for p in sub_parts if len(p.strip()) > 10])
            else:
                items.append(part)
        
        # 去重
        seen = set()
        result = []
        for item in items:
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        
        return result[:30]


# ====================== Record Builder ======================

def build_record(parser: FormatAwareDocParser) -> Dict:
    """构建标准化记录"""
    extractor = StateBasedExtractor(parser)
    
    trial_id = extractor.extract_trial_id()
    public_title, scientific_title = extractor.extract_titles()
    
    record = {
        "trial_id": trial_id,
        "background": {
            "public_title": public_title,
            "scientific_title": scientific_title,
            "conditions": extractor.extract_conditions(),
            "interventions": extractor.extract_interventions(),
            "brief_summary": extractor.extract_brief_summary(),
        },
        "criteria": {
            "inclusion": extractor.extract_criteria_format_aware('inclusion'),
            "exclusion": extractor.extract_criteria_format_aware('exclusion'),
        },
        "meta": {
            "phase": extractor.extract_phase(),
            "raw_file": os.path.basename(parser.path),
            "parser_mode": "DOM" if parser.root is not None else "REGEX_FALLBACK"
        }
    }
    
    return record


# ====================== Quality Assessment ======================

def assess_quality_v4(record: Dict) -> Tuple[bool, List[str]]:
    """V4质量评估（与V3相同，保持一致性）"""
    issues = []
    
    trial_id = record.get("trial_id", "").strip()
    bg = record.get("background", {})
    cr = record.get("criteria", {})
    
    # 硬性要求：trial_id
    if not trial_id or not trial_id.startswith("CTR"):
        issues.append("missing_or_invalid_trial_id")
        return False, issues
    
    # 至少有背景信息
    has_any_bg = any([
        bg.get("public_title"),
        bg.get("scientific_title"),
        bg.get("conditions"),
        bg.get("brief_summary")
    ])
    
    if not has_any_bg:
        issues.append("no_background_info")
    
    # 至少有入排标准
    inc_count = len(cr.get("inclusion", []))
    exc_count = len(cr.get("exclusion", []))
    
    if inc_count == 0 and exc_count == 0:
        issues.append("no_eligibility_criteria")
    
    # 警告
    if inc_count == 0:
        issues.append("warning_no_inclusion")
    if exc_count == 0:
        issues.append("warning_no_exclusion")
    
    # Clean判定
    is_clean = not any(issue.startswith("missing") or issue.startswith("no_") 
                       for issue in issues if "warning" not in issue)
    
    return is_clean, issues


# ====================== Main Processing ======================

def process_file(file_path: str) -> Tuple[Optional[Dict], List[str]]:
    """处理单个文件"""
    try:
        parser = FormatAwareDocParser(file_path)
        
        if not parser.parse():
            return None, ["parse_failed"]
        
        record = build_record(parser)
        
        return record, []
        
    except Exception as e:
        return None, [f"error: {str(e)}"]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    print("=" * 80)
    print("TrialGPT-China 数据清洗脚本 - V4 格式感知版")
    print("Structure + Format Aware Parsing (SIGIR-Ready)")
    print("=" * 80)
    
    # 准备目录
    ensure_dir(OUT_DIR)
    clean_dir = os.path.join(OUT_DIR, "clean_parts")
    dirty_dir = os.path.join(OUT_DIR, "dirty")
    ensure_dir(clean_dir)
    ensure_dir(dirty_dir)
    
    # 输出文件
    dirty_path = os.path.join(dirty_dir, "trials_dirty.jsonl")
    failed_path = os.path.join(OUT_DIR, "failed_files.txt")
    report_path = os.path.join(OUT_DIR, "clean_report.json")
    sample_path = os.path.join(OUT_DIR, "sample_clean.jsonl")
    
    # 统计
    stats = {
        "total_files": 0,
        "parsed_ok": 0,
        "clean_count": 0,
        "dirty_count": 0,
        "fail_count": 0,
        "dom_mode_count": 0,
        "regex_mode_count": 0,
        "quality_issues": {},
    }
    
    sample_records = []
    
    # 分片管理
    shard_idx = 0
    shard_count = 0
    shard_file = None
    
    def open_new_shard():
        nonlocal shard_idx, shard_count, shard_file
        if shard_file:
            shard_file.close()
        shard_path = os.path.join(clean_dir, f"trials_clean_part_{shard_idx:03d}.jsonl")
        shard_file = open(shard_path, "w", encoding="utf-8")
        shard_count = 0
        shard_idx += 1
    
    open_new_shard()
    
    with open(dirty_path, "w", encoding="utf-8") as f_dirty, \
         open(failed_path, "w", encoding="utf-8") as f_failed:
        
        # 遍历文件
        for root, _, files in os.walk(ROOT_DIR):
            for fname in files:
                if not fname.lower().endswith(".doc"):
                    continue
                
                stats["total_files"] += 1
                file_path = os.path.join(root, fname)
                
                # 进度
                if stats["total_files"] % 500 == 0:
                    current_rate = (stats["clean_count"] / stats["total_files"] * 100) if stats["total_files"] > 0 else 0
                    dom_rate = (stats["dom_mode_count"] / stats["parsed_ok"] * 100) if stats["parsed_ok"] > 0 else 0
                    print(f"[进度] 已处理: {stats['total_files']} | "
                          f"Clean: {stats['clean_count']} ({current_rate:.1f}%) | "
                          f"DOM模式: {stats['dom_mode_count']} ({dom_rate:.1f}%)")
                
                # 处理
                record, errors = process_file(file_path)
                
                if record is None:
                    stats["fail_count"] += 1
                    f_failed.write(f"{file_path}\t{';'.join(errors)}\n")
                    continue
                
                stats["parsed_ok"] += 1
                
                # 统计解析模式
                parser_mode = record.get("meta", {}).get("parser_mode", "UNKNOWN")
                if parser_mode == "DOM":
                    stats["dom_mode_count"] += 1
                else:
                    stats["regex_mode_count"] += 1
                
                # 质量评估
                is_clean, issues = assess_quality_v4(record)
                
                if is_clean:
                    # Clean数据
                    shard_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["clean_count"] += 1
                    shard_count += 1
                    
                    # 抽样
                    if len(sample_records) < SAMPLE_N and random.random() < 0.03:
                        sample_records.append(record)
                    
                    # 分片
                    if shard_count >= SHARD_SIZE:
                        open_new_shard()
                else:
                    # Dirty数据
                    record["_quality_issues"] = issues
                    f_dirty.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["dirty_count"] += 1
                    
                    for issue in issues:
                        stats["quality_issues"][issue] = stats["quality_issues"].get(issue, 0) + 1
    
    # 关闭文件
    if shard_file:
        shard_file.close()
    
    # 样本
    with open(sample_path, "w", encoding="utf-8") as f_sample:
        for rec in sample_records:
            f_sample.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # 报告
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "V4 (Structure + Format Aware)",
        "config": {
            "root_dir": ROOT_DIR,
            "out_dir": OUT_DIR,
            "shard_size": SHARD_SIZE,
        },
        "statistics": stats,
        "clean_rate": stats["clean_count"] / stats["total_files"] if stats["total_files"] > 0 else 0,
        "parse_success_rate": stats["parsed_ok"] / stats["total_files"] if stats["total_files"] > 0 else 0,
        "dom_success_rate": stats["dom_mode_count"] / stats["parsed_ok"] if stats["parsed_ok"] > 0 else 0,
        "notes": "V4: Structure-Aware (DOM) + Format-Aware (Bold/FontSize) + State Machine"
    }
    
    with open(report_path, "w", encoding="utf-8") as f_report:
        json.dump(report, f_report, ensure_ascii=False, indent=2)
    
    # 总结
    print("\n" + "=" * 80)
    print("✅ 数据清洗完成！")
    print("=" * 80)
    print(f"📊 统计信息:")
    print(f"  - 总文件数: {stats['total_files']}")
    print(f"  - 解析成功: {stats['parsed_ok']} ({report['parse_success_rate']:.1%})")
    print(f"  - Clean数据: {stats['clean_count']} ({report['clean_rate']:.1%})")
    print(f"  - Dirty数据: {stats['dirty_count']}")
    print(f"  - 失败文件: {stats['fail_count']}")
    print(f"\n🔍 解析模式统计:")
    print(f"  - DOM模式: {stats['dom_mode_count']} ({report['dom_success_rate']:.1%})")
    print(f"  - REGEX回退: {stats['regex_mode_count']}")
    
    if stats['quality_issues']:
        print(f"\n📋 主要质量问题:")
        for issue, count in sorted(stats['quality_issues'].items(), key=lambda x: -x[1])[:5]:
            print(f"  - {issue}: {count}")
    
    print(f"\n📁 输出目录:")
    print(f"  - Clean分片: {clean_dir}")
    print(f"  - Dirty数据: {dirty_dir}")
    print(f"  - 统计报告: {report_path}")
    print(f"  - 样本文件: {sample_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
