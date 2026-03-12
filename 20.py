import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
import sys
from pathlib import Path
import unicodedata
import math
import itertools

# ==================== 設定エリア ====================
# 比較するモデルのリスト（使用するモデルのコメントを外す）
COMPARE_MODELS = [
    "hfl_chinese-macbert-base",
    "hfl_chinese-macbert-large",
    "hfl_chinese-bert-wwm-ext",
    "hfl_chinese-roberta-wwm-ext",
    "hfl_chinese-electra-180g-base-generator",
    "hfl_chinese-electra-180g-large-generator",
### "ckiplab_bert-base-chinese",
    "ethanyt_guwenbert-base",
    "bert-base-chinese"
]

# モデル名の略称
MODEL_ABBREVIATIONS = {
    "hfl_chinese-macbert-base": "M-b",    # MacBERT Base（推奨）- 中国語マスク言語モデル
    "hfl_chinese-macbert-large": "M-l",   # MacBERT Large - 高精度版
    "hfl_chinese-bert-wwm-ext": "B-w",    # BERT WWM (Whole Word Masking)
    "hfl_chinese-roberta-wwm-ext": "R-w", # RoBERTa WWM Base
    "hfl_chinese-electra-180g-base-generator": "E-b",  # ELECTRA Generator Base
    "hfl_chinese-electra-180g-large-generator": "E-l", # ELECTRA Generator Large
### "ckiplab_bert-base-chinese": "C-b",   # CKIP BERT - 台湾中国語対応  GPL-3.0 licenseのため利用不可
    "ethanyt_guwenbert-base": "G-b",      # GuwenBERT - 古文中国語専用
    "bert-base-chinese": "B-b"            # BERT Base Chinese - Google公式
}

# 候補数
TOP_K = 10

# 文脈考慮モード（True: 行全体を考慮, False: 1文のみ考慮）
USE_FULL_LINE_CONTEXT = True

# 確率表示（True: 表示, False: 非表示）
SHOW_PROBABILITY = True

# 確率の小数点以下桁数
PROB_DECIMALS_DISPLAY = 4  # 画面表示用
PROB_DECIMALS_FILE = 4     # ファイル出力用

# モデルの最大トークン数
MAX_TOKENS = 512

# 文ペア評価（候補プール）設定
CANDIDATE_POOL_TOP_K = 20 # 計算コスト注意: 候補数×行数×ウィンドウ評価。遅い場合は20以下に調整 30
PLL_CONTEXT_WINDOW = 50   # MASK位置を中心に前後50トークンを評価する 遅い場合は30に調整
                
# ==================== アンサンブルスコアリング設定 ====================
MIN_PROBABILITY_THRESHOLD = 0.015  # 1.5%未満は除外
MAX_RANK_TO_CONSIDER = 10  # 評価する最大順位
MIN_SUPPORT = 2  # 最小支持モデル数

# 使用するスコア計算方式
RANK_SCORE_METHOD = 'exponential'  # 'exponential', 'logarithmic', 'linear', 'inverse'

# アンサンブル結果の表示候補数
ENSEMBLE_TOP_N = 10

# 使用する方式の選択
ENSEMBLE_METHODS = ["rank", "probability", "hybrid"]
# =====================================================================


def display_ensemble_comparison(ensemble_by_method, process_count,
                                mask_type=None, group_lines=None,
                                original_text=None,
                                mask_count=1,
                                f=None):
    """3方式のアンサンブル結果を横並びで表示"""
    
    COL_WIDTH = 18  # 列幅（全角考慮）— 画面表示用
    methods = ["rank", "probability", "hybrid"]
    
    def pad_to_width(s, width):
        w = 0
        for ch in s:
            if unicodedata.east_asian_width(ch) in ('F', 'W', 'A'):
                w += 2
            else:
                w += 1
        return s + " " * max(0, width - w)
    
    def build_cell(item, method):
        if not item:
            return ""
        token = item['token']
        score = item['score']
        if method == "rank":
            return f"{token} ({score:.2f})"
        elif method == "probability":
            return f"{token} ({score:.4f})"
        elif method == "hybrid":
            return f"{token} ({score:.3f})"
        return f"{token} ({score})"

    # ヘッダー部（共通）
    header_lines = []
    header_lines.append("=" * 80)
    header_lines.append(f"{process_count}番目")
    header_lines.append("-" * 80)

    if mask_type and group_lines:
        if mask_type == 'MASK':
            header_lines.append("通常MASKグループ:")
        else:
            header_lines.append(f"MASK_{mask_type}グループ:")
        for gline_num, gline_text in group_lines:
            header_lines.append(f"  行{gline_num}: {gline_text}")
    elif original_text:
        header_lines.append(original_text)

    header_lines.append("-" * 80)

    for mask_idx in range(mask_count):
        if mask_type and mask_type != 'MASK':
            header_lines.append(f"[MASK_{mask_type}]:")
        else:
            header_lines.append(f"[MASK]{mask_idx + 1}:")

        # 各方式のこのmask_idxの結果を取得
        method_results = []
        for method in methods:
            results = ensemble_by_method.get(method, [])
            if mask_idx < len(results):
                method_results.append(results[mask_idx])
            else:
                method_results.append([])

        max_rank = max((len(r) for r in method_results), default=0)

        if f is None:
            # === 画面表示：スペースパディング ===
            header = "      " + "".join([pad_to_width(m, COL_WIDTH) for m in methods])
            header_lines.append(header)

            for rank in range(max_rank):
                row = f"{rank+1:>2}位:  "
                for mi, method in enumerate(methods):
                    items = method_results[mi]
                    if rank < len(items):
                        row += pad_to_width(build_cell(items[rank], method), COL_WIDTH)
                    else:
                        row += " " * COL_WIDTH
                header_lines.append(row)
        else:
            # === ファイル出力：タブ区切り ===
            header = "\t".join([""] + methods)
            header_lines.append(header)

            for rank in range(max_rank):
                cells = [f"{rank+1}位"]
                for mi, method in enumerate(methods):
                    items = method_results[mi]
                    if rank < len(items):
                        cells.append(build_cell(items[rank], method))
                    else:
                        cells.append("")
                header_lines.append("\t".join(cells))

        header_lines.append("")

    text = "\n".join(header_lines)
    if f is None:
        print("\n" + text)
    else:
        f.write(text + "\n")
        
        
def get_rank_score(rank, method=RANK_SCORE_METHOD):
    """順位に応じたスコアを計算"""
    if rank <= 0 or rank > MAX_RANK_TO_CONSIDER:
        return 0.0
    
    if method == 'exponential':
        return 10.0 * (0.75 ** (rank - 1))
    elif method == 'logarithmic':
        return 10.0 / math.log2(rank + 1)
    elif method == 'linear':
        return float(MAX_RANK_TO_CONSIDER + 1 - rank)
    elif method == 'inverse':
        return 10.0 / rank
    else:
        return 0.0


# アンサンブル表示共通化
def build_ensemble_output(ensemble_results,
                          line_num,
                          process_count,
                          original_text,
                          mask_type=None,
                          group_lines=None,
                          method=None):

    lines = []

    lines.append("=" * 80)
    lines.append(f"{process_count}番目")
    
    if mask_type and group_lines:
        if mask_type == 'MASK':
            lines.append("通常MASKグループ:")
        else:
            lines.append(f"MASK_{mask_type}グループ:")
        
        for gline_num, gline_text in group_lines:
            lines.append(f"  行{gline_num}: {gline_text}")
    else:
        lines.append(original_text)
    
    lines.append("-" * 80)

    # 各MASKの結果
    for idx, mask_result in enumerate(ensemble_results):
        if mask_type and mask_type != 'MASK':
            lines.append(f"[MASK_{mask_type}]:")
        else:
            lines.append(f"[MASK]{idx+1}:")

        for rank, item in enumerate(mask_result[:ENSEMBLE_TOP_N], 1):
            if isinstance(item, dict):
                token = item['token']
                score = item['score']
                support = item.get('support', 0)
                details = item.get('details', [])
                
                detail_strs = []
                for model_name, model_rank, prob in details:
                    abbr = MODEL_ABBREVIATIONS.get(model_name, model_name)
                    detail_strs.append(f"{abbr}: {model_rank}位({prob:.4f})")
                
                detail_text = ", ".join(detail_strs)
                
                if method == "rank":
                    lines.append(f"  {rank:>2}位: {token} ({score:.2f}点) ← {support}モデルが支持（詳細: {detail_text}）")
                elif method == "probability":
                    lines.append(f"  {rank:>2}位: {token} ({score:.4f}) ← {support}モデルが支持（詳細: {detail_text}）")
                elif method == "hybrid":
                    lines.append(f"  {rank:>2}位: {token} ({score:.3f}点) ← {support}モデルが支持（詳細: {detail_text}）")
            else:
                token, score = item
                lines.append(f"  {rank:>2}位: {token} ({score:.6f})")

        lines.append("")

    return "\n".join(lines)


# 画面表示
def display_ensemble_results(**kwargs):
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")
    
    output = build_ensemble_output(
        ensemble_results,
        kwargs["line_num"],
        kwargs["process_count"],
        kwargs["original_text"],
        mask_type=mask_type,
        group_lines=group_lines,
        method=method
    )
    print(output)


# ファイル保存
def save_ensemble_results(output_file, **kwargs):
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")

    output = build_ensemble_output(
        ensemble_results,
        kwargs["line_num"],
        kwargs["process_count"],
        kwargs["original_text"],
        mask_type=mask_type,
        group_lines=group_lines,
        method=method
    )

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(output + "\n")


def display_comparison_results(all_model_predictions, model_names, line_text,
                               line_num=None, process_count=None,
                               show_prob=True, decimals=4,
                               mask_type=None, group_lines=None,
                               mask_index=None,
                               show_header=True,
                               f=None):

    COL_WIDTH = 14  # 列幅（全角考慮）— 画面表示用
    
    def pad_to_width(s, width):
        w = 0
        for ch in s:
            if unicodedata.east_asian_width(ch) in ('F', 'W', 'A'):
                w += 2
            else:
                w += 1
        return s + " " * max(0, width - w)

    # ヘッダー部（画面・ファイル共通）
    header_lines = []
    
    if show_header:
        header_lines.append("=" * 80)
        header_lines.append(f"{process_count}番目")
        header_lines.append("-" * 80)

        if mask_type and group_lines:
            if mask_type == 'MASK':
                header_lines.append("通常MASKグループ:")
            else:
                header_lines.append(f"MASK_{mask_type}グループ:")

            for gline_num, gline_text in group_lines:
                header_lines.append(f"  行{gline_num}: {gline_text}")

            header_lines.append("-" * 80)
        else:
            header_lines.append(line_text)
            header_lines.append("-" * 80)

    # MASKインデックス表示
    if mask_index is not None:
        if mask_type and mask_type != 'MASK':
            header_lines.append(f"[MASK_{mask_type}]:")
        else:
            header_lines.append(f"[MASK]{mask_index + 1}:")

    preds_list = [p if isinstance(p, list) else [] for p in all_model_predictions]
    max_rank = max((len(p) for p in preds_list), default=0)

    if max_rank == 0:
        header_lines.append("予測結果なし")
        text = "\n".join(header_lines)
        print(text)
        if f is not None:
            f.write(text + "\n")
        return

    # データ部を画面用・ファイル用それぞれ構築
    def build_cell(item, show_prob, decimals):
        if isinstance(item, (list, tuple)) and len(item) == 2:
            token, prob = item
            if show_prob:
                return f"{token} ({prob:.{decimals}f})"
            else:
                return f"{token}"
        else:
            return f"{item}"

    if f is None:
        # === 画面表示：スペースパディング ===
        lines = list(header_lines)
        header = "      " + "".join([pad_to_width(name, COL_WIDTH) for name in model_names])
        lines.append(header)

        for rank in range(max_rank):
            row = f"{rank+1:>2}位:  "
            for model_preds in preds_list:
                if rank < len(model_preds):
                    row += pad_to_width(build_cell(model_preds[rank], show_prob, decimals), COL_WIDTH)
                else:
                    row += " " * COL_WIDTH
            lines.append(row)

        print("\n" + "\n".join(lines))
    else:
        # === ファイル出力：タブ区切り ===
        lines = list(header_lines)
        header = "\t".join([""] + list(model_names))
        lines.append(header)

        for rank in range(max_rank):
            cells = [f"{rank+1:>2}位"]
            for model_preds in preds_list:
                if rank < len(model_preds):
                    cells.append(build_cell(model_preds[rank], show_prob, decimals))
                else:
                    cells.append("")
            lines.append("\t".join(cells))

        f.write("\n".join(lines) + "\n\n")

class ChineseMaskCompletion:
    def __init__(self, model_path, top_k=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path).to(self.device)
        self.model.eval()
        self.top_k = top_k
        self.mask_token = self.tokenizer.mask_token
    
    
    def split_sentences(self, text):
        """文章を句点で分割"""
        sentences = re.split(r'([。！？!?])', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]
    
    # 許可する中国語句読点・記号
    ALLOWED_PUNCT = {
        "，", "。", "！", "？", "、", "：", "；",
        "\u201c", "\u201d",  # ""
        "\u2018", "\u2019",  # ''
        "「", "」", "『", "』",
        "（", "）", "《", "》",
        "—", "…", "·",
    }

    @staticmethod
    def is_valid_chinese_token(token: str) -> bool:
        if not token:
            return False

        if token in {"<UNK>", "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}:
            return False
    
        if token.startswith("[") and token.endswith("]"):
            return False
    
        if token.startswith("##"):
            return False

        if len(token) > 2:
            return False

        ch = token[0]

        # 許可された句読点
        if ch in ChineseMaskCompletion.ALLOWED_PUNCT:
            return True

        # 絵文字・記号除外（句読点チェックの後）
        cat = unicodedata.category(ch)
        if cat.startswith("S"):
            return False

        # CJK統合漢字
        if not (
            "\u4e00" <= ch <= "\u9fff" or
            "\u3400" <= ch <= "\u4dbf" or
            "\u20000" <= ch <= "\u2a6df" or
            "\u2a700" <= ch <= "\u2b73f"
        ):
            return False

        return True
    

    def evaluate_mlm_score_fast(self, text, skip_positions=None):
        """文全体のMLMスコアを計算"""
        if skip_positions is None:
            skip_positions = set()
    
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=MAX_TOKENS).to(self.device)
        input_ids = inputs["input_ids"].clone()
    
        special_ids = set([
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id
        ])
    
        eval_positions = []
        for i in range(input_ids.size(1)):
            token_id = input_ids[0, i].item()
            if i in skip_positions or token_id in special_ids:
                continue
            eval_positions.append(i)
    
        if not eval_positions:
            return 0.0
    
        batch_size = len(eval_positions)
        MAX_BATCH = 64  # GPUメモリに応じて調整
        
        all_logits = []
        for batch_start in range(0, batch_size, MAX_BATCH):
            batch_end = min(batch_start + MAX_BATCH, batch_size)
            batch_positions = eval_positions[batch_start:batch_end]
            
            mini_batch = input_ids.repeat(len(batch_positions), 1)
            for idx, pos in enumerate(batch_positions):
                mini_batch[idx, pos] = self.tokenizer.mask_token_id
            
            with torch.no_grad():
                outputs = self.model(mini_batch)
                all_logits.append(outputs.logits.cpu())
        
        logits = torch.cat(all_logits, dim=0)
    
        total_score = 0.0
        for idx, pos in enumerate(eval_positions):
            original_token = input_ids[0, pos].item()
            token_logits = logits[idx, pos]
            log_probs = torch.log_softmax(token_logits, dim=0)
            total_score += log_probs[original_token].item()
    
        return total_score
    
    def get_context_window(self, text, use_full_context=True):
        """文脈ウィンドウを取得（全MASK位置を含む、512トークン制約考慮）"""
        if not use_full_context:
            sentences = self.split_sentences(text)
            # 全MASKを含む文を収集
            mask_sentences = [s for s in sentences if '[MASK]' in s]
            if mask_sentences:
                return ''.join(mask_sentences)
            return text
        
        # 512トークン制約チェック
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= MAX_TOKENS:
            return text
        
        sentences = self.split_sentences(text)
        
        # 全MASKを含む文のインデックスを取得
        mask_indices = [i for i, s in enumerate(sentences) if '[MASK]' in s]
        
        if not mask_indices:
            decoded = self.tokenizer.decode(tokens[:MAX_TOKENS], skip_special_tokens=True)
            return decoded
        
        # 最初のMASK〜最後のMASKを含む範囲を必須コアにする
        first_mask = mask_indices[0]
        last_mask = mask_indices[-1]
        
        core = list(range(first_mask, last_mask + 1))
        context_indices = list(core)
        
        # コアだけで512超なら、コアのみ返す（それでも超える場合はトークン切り詰め）
        core_text = ''.join(sentences[i] for i in context_indices)
        core_tokens = len(self.tokenizer.encode(core_text, add_special_tokens=True))
        if core_tokens >= MAX_TOKENS:
            encoded = self.tokenizer.encode(core_text, add_special_tokens=True,
                                             truncation=True, max_length=MAX_TOKENS)
            return self.tokenizer.decode(encoded, skip_special_tokens=True)
        
        # コアから前後に拡張
        left = first_mask - 1
        right = last_mask + 1
        
        while left >= 0 or right < len(sentences):
            current_text = ''.join(sentences[i] for i in sorted(context_indices))
            current_tokens = len(self.tokenizer.encode(current_text, add_special_tokens=True))
            
            if current_tokens >= MAX_TOKENS:
                break
            
            added = False
            if left >= 0:
                test_indices = sorted(context_indices + [left])
                test_text = ''.join(sentences[i] for i in test_indices)
                if len(self.tokenizer.encode(test_text, add_special_tokens=True)) <= MAX_TOKENS:
                    context_indices.append(left)
                    left -= 1
                    added = True
                else:
                    left = -1
            
            if right < len(sentences):
                test_indices = sorted(context_indices + [right])
                test_text = ''.join(sentences[i] for i in test_indices)
                if len(self.tokenizer.encode(test_text, add_special_tokens=True)) <= MAX_TOKENS:
                    context_indices.append(right)
                    right += 1
                    added = True
                else:
                    right = len(sentences)
            
            if not added:
                break
        
        return ''.join(sentences[i] for i in sorted(context_indices))
    
    def predict_masks(self, text, use_full_context=True):
        """複数の[MASK]を予測"""
        context_text = self.get_context_window(text, use_full_context)

        inputs = self.tokenizer(context_text, return_tensors="pt",
                                truncation=True, max_length=MAX_TOKENS).to(self.device)

        mask_token_indices = (inputs.input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_token_indices) == 0:
            return []

        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        results = []
        for mask_idx, token_idx in enumerate(mask_token_indices):
            mask_logits = predictions[0, token_idx]
            probs = torch.softmax(mask_logits, dim=0)
        
            vocab_size = len(probs)
            OVERGEN_K = min(self.top_k * 10, vocab_size)
        
            top_probs, top_indices = torch.topk(probs, OVERGEN_K)

            candidates = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.convert_ids_to_tokens(int(idx))
        
                if not self.is_valid_chinese_token(token):
                    continue
            
                candidates.append({
                    'token': token,
                    'probability': prob.item()
                })
        
                if len(candidates) >= self.top_k:
                    break
    
            results.append({
                'position': mask_idx,
                'candidates': candidates
            })

        return results


def predict_masks_per_window(model, text, use_full_context=True):
    """
    各MASK位置ごとに個別のコンテキストウィンドウで予測
    512トークン以上離れたMASK対応
    """
    inputs = model.tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"][0]
    mask_token_id = model.tokenizer.mask_token_id

    # ★修正：tensorで直接検出
    mask_positions_in_tokens = (input_ids == mask_token_id).nonzero(as_tuple=True)[0].tolist()

    if not mask_positions_in_tokens:
        return []

    full_tokens = input_ids.tolist()
    
    # 全MASKが512トークン内に収まるか確認
    total_span = mask_positions_in_tokens[-1] - mask_positions_in_tokens[0] + 1
    if total_span + 2 <= MAX_TOKENS:
        return model.predict_masks(text, use_full_context)
    
    # MASKが離れすぎ → 各MASK位置ごとに個別ウィンドウ
    results = []
    half_window = (MAX_TOKENS - 2) // 2

    for mask_idx, mask_pos in enumerate(mask_positions_in_tokens):
        start = max(0, mask_pos - half_window)
        end = min(len(full_tokens), mask_pos + half_window + 1)
        
        if end - start > MAX_TOKENS - 2:
            if mask_pos - start > half_window:
                start = mask_pos - half_window
            end = start + MAX_TOKENS - 2
        
        window_tokens = full_tokens[start:end]
        local_mask_pos = mask_pos - start
        
        input_ids = torch.tensor(
            [[model.tokenizer.cls_token_id] + window_tokens + [model.tokenizer.sep_token_id]]
        ).to(model.device)
        
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        mask_logits = logits[0, local_mask_pos + 1]
        probs = torch.softmax(mask_logits, dim=0)
        
        vocab_size = len(probs)
        OVERGEN_K = min(model.top_k * 10, vocab_size)
        top_probs, top_indices = torch.topk(probs, OVERGEN_K)
        
        candidates = []
        for prob, idx in zip(top_probs, top_indices):
            token = model.tokenizer.convert_ids_to_tokens(int(idx))
            if model.is_valid_chinese_token(token):
                candidates.append({
                    'token': token,
                    'probability': prob.item()
                })
                if len(candidates) >= model.top_k:
                    break
        
        results.append({
            'position': mask_idx,
            'candidates': candidates
        })
    
    return results


def evaluate_consecutive_masks(model, text, top_n=20, use_full_context=True): 
    """
    連続MASKの組み合わせを評価
    例: [MASK][MASK] → 上位N×Nの組み合わせからPLLで再スコア
    """
    context_text = model.get_context_window(text, use_full_context)
    inputs = model.tokenizer(context_text, return_tensors="pt",
                             truncation=True, max_length=MAX_TOKENS).to(model.device)
    
    mask_positions = (inputs.input_ids == model.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist()
    
    if len(mask_positions) < 2:
        return None

    # 連続グループを検出
    groups = []
    current_group = [mask_positions[0]]
    for i in range(1, len(mask_positions)):
        if mask_positions[i] == mask_positions[i-1] + 1:
            current_group.append(mask_positions[i])
        else:
            groups.append(current_group)
            current_group = [mask_positions[i]]
    groups.append(current_group)
    
    if all(len(g) == 1 for g in groups):
        return None
    
    with torch.no_grad():
        outputs = model.model(**inputs)
        logits = outputs.logits
    
    group_results = []
    
    for group in groups:
        if len(group) == 1:
            pos = group[0]
            probs = torch.softmax(logits[0, pos], dim=0)
            vocab_size = len(probs)
            top_probs, top_indices = torch.topk(probs, min(top_n * 5, vocab_size))
            
            candidates = []
            for prob, idx in zip(top_probs, top_indices):
                token = model.tokenizer.convert_ids_to_tokens(int(idx))
                if model.is_valid_chinese_token(token):
                    candidates.append({
                        'tokens': [token],
                        'score': 0.0,
                        'probability': prob.item()
                    })
                    if len(candidates) >= top_n:
                        break
            group_results.append(candidates)
            continue
        
        # 連続MASK: 各位置の上位N候補を取得
        position_candidates = []
        for pos in group:
            probs = torch.softmax(logits[0, pos], dim=0)
            vocab_size = len(probs)
            top_probs, top_indices = torch.topk(probs, min(top_n * 5, vocab_size))
            
            pos_cands = []
            for prob, idx in zip(top_probs, top_indices):
                token = model.tokenizer.convert_ids_to_tokens(int(idx))
                if model.is_valid_chinese_token(token):
                    pos_cands.append((token, prob.item(), int(idx)))
                    if len(pos_cands) >= top_n:
                        break
            position_candidates.append(pos_cands)
        
        # 組み合わせ探索
        if len(group) == 2:
            combos = [
                (c1, c2)
                for c1 in position_candidates[0]
                for c2 in position_candidates[1]
            ]
        elif len(group) == 3:
            limit = min(top_n, 10)  # 計算コスト注意: 10^3=1000通り。遅い場合は8以下に調整
            combos = [
                (c1, c2, c3)
                for c1 in position_candidates[0][:limit]
                for c2 in position_candidates[1][:limit]
                for c3 in position_candidates[2][:limit]
            ]
        elif len(group) <= 4:
            limit = 5          # 計算コスト注意: 5^4=625通り。遅い場合はlimit=3〜4に調整
            trimmed = [pc[:limit] for pc in position_candidates]
            combos = list(itertools.product(*trimmed))
        else:
            # 連続5以上はスキップ
            print(f"  警告: 連続{len(group)}MASKは計算コストが高いためスキップ")
            group_results.append([])
            continue
        
        # 各組み合わせをPLLで評価
        scored_combos = []
        for combo in combos:
            test_ids = inputs.input_ids.clone()
            for i, (token, prob, token_id) in enumerate(combo):
                test_ids[0, group[i]] = token_id
            
            with torch.no_grad():
                test_outputs = model.model(test_ids)
                test_logits = test_outputs.logits
            
            # 注意: 擬似PLL（他MASKも埋めた状態で評価）。
            # 完全PLLでは各位置を単独再マスクすべきだが、
            # 補完用途では精度的に十分。
            combo_score = 0.0
            for i, (token, prob, token_id) in enumerate(combo):
                pos = group[i]
                pos_probs = torch.softmax(test_logits[0, pos], dim=0)
                combo_score += math.log(pos_probs[token_id].item() + 1e-12)
            
            indep_score = sum(math.log(c[1] + 1e-12) for c in combo)
            final_score = 0.6 * combo_score + 0.4 * indep_score
            
            scored_combos.append({
                'tokens': [c[0] for c in combo],
                'score': final_score,
                'probability': math.exp(final_score)
            })
        
        scored_combos.sort(key=lambda x: -x['score'])
        
        if scored_combos:
            max_prob = max(x['probability'] for x in scored_combos)
            for item in scored_combos:
                item['probability'] = item['probability'] / max_prob if max_prob > 0 else 0.0
        
        group_results.append(scored_combos[:TOP_K])
    
    return group_results


def calculate_ensemble_scores_rank(all_model_predictions, model_names):
    """方式A: 順位ベーススコアリング"""
    if not model_names or not all_model_predictions:
        return []
    
    pred_lengths = [len(preds) for preds in all_model_predictions.values() if len(preds) > 0]
    num_masks = max(pred_lengths) if pred_lengths else 0
    ensemble_results = []
    
    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions.get(model_name, [])
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                if prob < MIN_PROBABILITY_THRESHOLD:
                    continue
                
                score = get_rank_score(rank, RANK_SCORE_METHOD)
                
                if token not in token_scores:
                    token_scores[token] = {"score": 0, "models": []}
                
                token_scores[token]["score"] += score
                token_scores[token]["models"].append((model_name, rank, prob))
        
        filtered_tokens = {
            token: data 
            for token, data in token_scores.items() 
            if len(data["models"]) >= MIN_SUPPORT
        }
        
        sorted_tokens = sorted(
            filtered_tokens.items(), 
            key=lambda x: (-x[1]["score"], x[0])
        )
        
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({
                "token": token,
                "score": data["score"],
                "support": len(data["models"]),
                "details": data["models"]
            })
        
        ensemble_results.append(mask_result)
    
    return ensemble_results


def calculate_ensemble_scores_probability(all_model_predictions, model_names):
    """方式B: 確率ベーススコアリング"""
    if not model_names or not all_model_predictions:
        return []
    
    pred_lengths = [len(preds) for preds in all_model_predictions.values() if len(preds) > 0]
    num_masks = max(pred_lengths) if pred_lengths else 0
    ensemble_results = []
    
    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions.get(model_name, [])
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                if prob < MIN_PROBABILITY_THRESHOLD:
                    continue
                
                if token not in token_scores:
                    token_scores[token] = {"probs": [], "models": []}
                
                token_scores[token]["probs"].append(prob)
                token_scores[token]["models"].append((model_name, rank, prob))
        
        for token in token_scores:
            avg_prob = sum(token_scores[token]["probs"]) / len(token_scores[token]["probs"])
            token_scores[token]["score"] = avg_prob
        
        filtered_tokens = {
            token: data 
            for token, data in token_scores.items() 
            if len(data["models"]) >= MIN_SUPPORT
        }
        
        sorted_tokens = sorted(
            filtered_tokens.items(),
            key=lambda x: (-x[1]["score"], x[0])
        )
        
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({
                "token": token,
                "score": data["score"],
                "support": len(data["models"]),
                "details": data["models"]
            })
        
        ensemble_results.append(mask_result)
    
    return ensemble_results


def calculate_ensemble_scores_hybrid(all_model_predictions, model_names):
    """方式C: ハイブリッドスコアリング"""
    if not model_names or not all_model_predictions:
        return []
    
    pred_lengths = [len(preds) for preds in all_model_predictions.values() if len(preds) > 0]
    num_masks = max(pred_lengths) if pred_lengths else 0
    ensemble_results = []
    
    for mask_idx in range(num_masks):
        token_scores = {}
        
        for model_name in model_names:
            predictions = all_model_predictions.get(model_name, [])
            if mask_idx >= len(predictions):
                continue
            
            candidates = predictions[mask_idx].get('candidates', [])
            for rank, candidate in enumerate(candidates, start=1):
                if rank > MAX_RANK_TO_CONSIDER:
                    break
                
                token = candidate['token']
                prob = candidate['probability']
                
                if prob < MIN_PROBABILITY_THRESHOLD:
                    continue
                
                rank_score = get_rank_score(rank, RANK_SCORE_METHOD)
                
                if token not in token_scores:
                    token_scores[token] = {"score": 0, "models": []}
                
                hybrid_score = rank_score * prob
                token_scores[token]["score"] += hybrid_score
                token_scores[token]["models"].append((model_name, rank, prob))
        
        filtered_tokens = {
            token: data 
            for token, data in token_scores.items() 
            if len(data["models"]) >= MIN_SUPPORT
        }
        
        sorted_tokens = sorted(
            filtered_tokens.items(),
            key=lambda x: (-x[1]["score"], x[0])
        )
        
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({
                "token": token,
                "score": data["score"],
                "support": len(data["models"]),
                "details": data["models"]
            })
        
        ensemble_results.append(mask_result)
    
    return ensemble_results


def main():
    if len(sys.argv) < 2:
        print("使い方: python script.py <入力テキストファイル>")
        print("例: python script.py input.txt")
        return
    
    input_file = sys.argv[1]
    input_path = Path(input_file)
    
    # ★修正: Pathで出力ファイル名を生成
    output_path = input_path.with_name(input_path.stem + "_out" + input_path.suffix)
    output_file = str(output_path)
    
    # 既存の出力ファイルを削除（初期化）
    if output_path.exists():
        output_path.unlink()
    
    # アンサンブル結果ファイルも削除
    for method in ENSEMBLE_METHODS:
        method_path = output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix)
        if method_path.exists():
            method_path.unlink()
    
    # アンサンブル横並びファイルも削除
    ensemble_compare_path = output_path.with_name(output_path.stem + "_ensemble" + output_path.suffix)
    if ensemble_compare_path.exists():
        ensemble_compare_path.unlink()
        
    # モデルをロード
    models = {}
    for model_name in COMPARE_MODELS:
        model_path = f"models/{model_name}"
        if not Path(model_path).exists():
            print(f"警告: モデルが見つかりません: {model_path} (スキップします)")
            continue
        models[model_name] = ChineseMaskCompletion(model_path, top_k=TOP_K)
    
    if len(models) < MIN_SUPPORT:
        print(f"警告: ロード済みモデル数({len(models)})がMIN_SUPPORT({MIN_SUPPORT})未満です。アンサンブル結果が空になる可能性があります。")
        
    if not models:
        print("エラー: 使用可能なモデルがありません")
        return
    
    model_names = list(models.keys())
    
    # 入力ファイル読み込み
    if not Path(input_file).exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # ===== ステップ1: 全行を読み込み、マスクタイプごとに分類 =====
    mask_groups = {}
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        original_line = line
        custom_mask_pattern = r'\[MASK_([A-Z0-9_]+)\]'
        custom_matches = re.findall(custom_mask_pattern, line)
        
        temp_line = re.sub(custom_mask_pattern, '##TEMP##', line)
        has_normal_mask = '[MASK]' in temp_line
        
        if custom_matches:
            unique_custom = list(dict.fromkeys(custom_matches))
            
            for mask_type in unique_custom:
                # 1. 対象タイプを一時プレースホルダーに
                processed_line = re.sub(rf'\[MASK_{mask_type}\]', '##TARGET_MASK##', original_line)
                # 2. 他のカスタムマスクを除去
                processed_line = re.sub(r'\[MASK_[A-Z0-9_]+\]', '', processed_line)
                # 3. 通常[MASK]も除去（カスタムマスク処理では対象外）
                processed_line = processed_line.replace('[MASK]', '')
                # 4. プレースホルダーを[MASK]に戻す
                processed_line = processed_line.replace('##TARGET_MASK##', '[MASK]')
                
                if mask_type not in mask_groups:
                    mask_groups[mask_type] = []
                mask_groups[mask_type].append({
                    'line_num': line_num,
                    'original': original_line,
                    'processed': processed_line
                })
        
        if has_normal_mask:
            processed_line = re.sub(custom_mask_pattern, '', original_line)
            
            if 'MASK' not in mask_groups:
                mask_groups['MASK'] = []
            mask_groups['MASK'].append({
                'line_num': line_num,
                'original': original_line,
                'processed': processed_line
            })
    
    # ===== ステップ2: グループごとに処理 =====
    all_results = []
    process_count = 0
    
    
    for mask_type, entries in mask_groups.items():
        # ============================================================
        # 通常MASKの処理ブロック
        # ============================================================
        if mask_type == 'MASK':
            eval_model_name = 'hfl_chinese-macbert-large'
            if eval_model_name not in models:
                eval_model_name = next(iter(models))
            eval_model = models[eval_model_name]
                
            for entry in entries:
                process_count += 1
                
                group_lines = [(entry['line_num'], entry['original'])]
                combined_original = entry['original']
                combined_processed = entry['processed']
                
                print(f"\n処理中: 通常MASK（単一行）: {process_count}番目")
                print(f"対象行: 行{entry['line_num']}")
                
                # 連続MASK検出・組み合わせ評価
                consecutive_results = None
                if combined_processed.count('[MASK]') >= 2:
                    consecutive_results = evaluate_consecutive_masks(
                        eval_model,
                        combined_processed,
                        top_n=20,                # 計算コスト注意: 遅い場合はtop_n = 15に調整
                        use_full_context=USE_FULL_LINE_CONTEXT
                    )
                    # evaluate_consecutive_masks内で連続判定済み（Noneが返れば非連続）
                    if consecutive_results:
                        print(f"  ★連続MASK検出: 組み合わせ評価実行")
                        for gi, group_cands in enumerate(consecutive_results):
                            if group_cands and len(group_cands[0].get('tokens', [])) >= 2:
                                print(f"    グループ{gi+1} (連続{len(group_cands[0]['tokens'])}字):")
                                for ri, cand in enumerate(group_cands[:5], 1):
                                    word = ''.join(cand['tokens'])
                                    print(f"      {ri}位: {word} ({cand['probability']:.4f})")
                
                # 各モデルで予測
                all_model_predictions = {}
                for model_name, model in models.items():
                    predictions = predict_masks_per_window(model, combined_processed, use_full_context=USE_FULL_LINE_CONTEXT)
                    all_model_predictions[model_name] = predictions
                
                # ★修正: 全MASKについて表示（mask_idxごと）
                pred_lengths = [len(preds) for preds in all_model_predictions.values() if len(preds) > 0]
                num_masks = max(pred_lengths) if pred_lengths else 0
                
                model_abbr = [MODEL_ABBREVIATIONS.get(name, name) for name in model_names]
                
                for mask_idx in range(num_masks):
                    predictions_for_display = []
                    for model_name in model_names:
                        if model_name in all_model_predictions:
                            preds = all_model_predictions[model_name]
                            if mask_idx < len(preds):
                                candidates = preds[mask_idx].get('candidates', [])
                                predictions_for_display.append([
                                    (c['token'], c['probability'])
                                    for c in candidates[:TOP_K]
                                ])
                            else:
                                predictions_for_display.append([])
                    
                    # 画面表示
                    display_comparison_results(
                        predictions_for_display,
                        model_abbr,
                        combined_original,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        show_prob=True,
                        decimals=PROB_DECIMALS_DISPLAY,
                        mask_type=mask_type,
                        group_lines=group_lines,
                        mask_index=mask_idx,
                        show_header=(mask_idx == 0),
                        f=None
                    )
                    
                    # ファイル出力
                    with open(output_file, 'a', encoding='utf-8') as f:
                        display_comparison_results(
                            predictions_for_display,
                            model_abbr,
                            combined_original,
                            line_num=entry['line_num'],
                            process_count=process_count,
                            show_prob=True,
                            decimals=PROB_DECIMALS_FILE,
                            mask_type=mask_type,
                            group_lines=group_lines,
                            mask_index=mask_idx,
                            show_header=(mask_idx == 0),
                            f=f
                        )
                    

                
                # アンサンブル計算（3方式）
                ensemble_by_method = {}
                for method in ENSEMBLE_METHODS:
                    if method == "rank":
                        ensemble_results = calculate_ensemble_scores_rank(all_model_predictions, model_names)
                    elif method == "probability":
                        ensemble_results = calculate_ensemble_scores_probability(all_model_predictions, model_names)
                    elif method == "hybrid":
                        ensemble_results = calculate_ensemble_scores_hybrid(all_model_predictions, model_names)
                    
                    ensemble_by_method[method] = ensemble_results
                    
                    display_ensemble_results(
                        ensemble_results=ensemble_results,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        original_text=entry['original'],
                        mask_type=mask_type,
                        group_lines=group_lines,
                        method=method
                    )
                    
                    output_file_method = str(output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix))
                    save_ensemble_results(
                        output_file=output_file_method,
                        ensemble_results=ensemble_results,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        original_text=entry['original'],
                        mask_type=mask_type,
                        group_lines=group_lines,
                        method=method
                    )
                
                # 3方式横並び出力
                display_ensemble_comparison(
                    ensemble_by_method, process_count,
                    mask_type=mask_type, group_lines=group_lines,
                    original_text=entry['original'],
                    mask_count=num_masks,
                    f=None
                )
                ensemble_compare_file = str(output_path.with_name(output_path.stem + "_ensemble" + output_path.suffix))
                with open(ensemble_compare_file, 'a', encoding='utf-8') as ef:
                    display_ensemble_comparison(
                        ensemble_by_method, process_count,
                        mask_type=mask_type, group_lines=group_lines,
                        original_text=entry['original'],
                        mask_count=num_masks,
                        f=ef
                    )
                
                all_results.append((entry['line_num'], process_count, combined_original, all_model_predictions, model_names))
        
        # ============================================================
        # カスタムマスクの処理ブロック
        # ============================================================
        else:
            eval_model_name = 'hfl_chinese-macbert-large'
            if eval_model_name not in models:
                eval_model_name = next(iter(models))
            eval_model = models[eval_model_name]
            
            process_count += 1

            line_nums = [entry['line_num'] for entry in entries]
            group_lines = [(entry['line_num'], entry['original']) for entry in entries]

            if len(entries) >= 2:
                print(f"\n処理中: MASK_{mask_type}グループ（{len(entries)}行連結）: {process_count}番目")
            else:
                print(f"\n処理中: MASK_{mask_type}（単一行）: {process_count}番目")
            print(f"対象行: " + ", ".join(f"行{num}" for num in line_nums))

            combined_original = '\n'.join(entry['original'] for entry in entries)
            combined_processed = '\n'.join(entry['processed'] for entry in entries)

            
            mask_counts = [entry['processed'].count('[MASK]') for entry in entries]
            total_masks = sum(mask_counts)

            if len(entries) >= 2 or total_masks >= 2:
            
                
                ### 文ペア評価用モデル ############
                # ===== ステップ1: 全モデルから候補プールを収集 =====
                all_mask_candidates = {}
                
                for entry in entries:
                    clean_processed = entry['processed']
                    
                    for model_name, model in models.items():
                        preds = model.predict_masks(clean_processed, use_full_context=USE_FULL_LINE_CONTEXT)
                        if not preds:
                            continue
                        
                        for cand in preds[0]['candidates'][:CANDIDATE_POOL_TOP_K]:
                            token = cand['token']
                            prob = cand['probability']
                            
                            if token not in all_mask_candidates:
                                all_mask_candidates[token] = {
                                    'prob_sum': 0.0,
                                    'count': 0,
                                    'details': []
                                }
                            
                            all_mask_candidates[token]['prob_sum'] += prob
                            all_mask_candidates[token]['count'] += 1
                            all_mask_candidates[token]['details'].append((model_name, prob))
                
                # ===== ステップ2: フィルタリング =====
                candidate_pool_filtered = {}
                for token, data in all_mask_candidates.items():
                    if not ChineseMaskCompletion.is_valid_chinese_token(token):
                        continue
                    avg_prob = data['prob_sum'] / data['count']
                    if avg_prob < 0.01:
                        continue
                    candidate_pool_filtered[token] = data
                
                if not candidate_pool_filtered:
                    candidate_pool_filtered = {
                        token: data
                        for token, data in all_mask_candidates.items()
                        if ChineseMaskCompletion.is_valid_chinese_token(token)
                    }
                
                
                # ===== ウィンドウPLL評価関数 =====
                def evaluate_window_pll(model, filled_text, mask_char_pos, window_size=PLL_CONTEXT_WINDOW):
                    use_offsets = getattr(model.tokenizer, "is_fast", False) and model.tokenizer.is_fast
                    inputs = model.tokenizer(filled_text, return_tensors="pt",
                                             truncation=True, max_length=MAX_TOKENS,
                                             return_offsets_mapping=use_offsets)
    
                    offsets = None
                    if "offset_mapping" in inputs:
                        offsets = inputs.pop("offset_mapping")[0].cpu().tolist()
    
                    inputs = inputs.to(model.device)
                    input_ids = inputs["input_ids"].clone()
                    seq_len = input_ids.size(1)
    
                    special_ids = set([
                        model.tokenizer.cls_token_id,
                        model.tokenizer.sep_token_id,
                        model.tokenizer.pad_token_id
                    ])
    
                    # MASK文字位置からトークン位置を特定（短文処理の前に移動）
                    mask_token_pos = None
                    if offsets is not None:
                        for tidx, (start, end) in enumerate(offsets):
                            if start <= mask_char_pos < end:
                                mask_token_pos = tidx
                                break
    
                    # offset_mapping が必須
                    if mask_token_pos is None:
                        print(f"警告: offset_mapping未対応。文字位置{mask_char_pos}のトークン位置特定に失敗")
                        return 0.0, 1  # ★不正なスコアを返さない
    
                    # 短い文はそのまま全評価（候補位置を除外）
                    if seq_len <= window_size * 2 + 2:
                        skip_positions = {mask_token_pos}  # ★追加
                        score = model.evaluate_mlm_score_fast(filled_text, skip_positions=skip_positions)  # ★修正
                        eval_count = sum(
                            1 for i in range(seq_len)
                            if input_ids[0, i].item() not in special_ids and i != mask_token_pos  # ★修正
                        )
                        return score, max(eval_count, 1)
    
                    # MASK位置を中心にウィンドウを設定
                    start = max(1, mask_token_pos - window_size)
                    end = min(seq_len - 1, mask_token_pos + window_size + 1)
    
                    eval_positions = []
                    for i in range(start, end):
                        token_id = input_ids[0, i].item()
                        if token_id in special_ids:
                            continue
                        if i == mask_token_pos:  # ★追加: 候補トークン位置をスキップ
                            continue
                        eval_positions.append(i)
    
                    if not eval_positions:
                        return 0.0, 1
    
                    MAX_BATCH = 64
                    all_logits = []
                    for batch_start in range(0, len(eval_positions), MAX_BATCH):
                        batch_end = min(batch_start + MAX_BATCH, len(eval_positions))
                        batch_positions = eval_positions[batch_start:batch_end]
        
                        mini_batch = input_ids.repeat(len(batch_positions), 1)
                        for idx, pos in enumerate(batch_positions):
                            mini_batch[idx, pos] = model.tokenizer.mask_token_id
        
                        with torch.no_grad():
                            outputs = model.model(mini_batch)
                            all_logits.append(outputs.logits.cpu())
    
                    logits = torch.cat(all_logits, dim=0)
    
                    total_score = 0.0
                    for idx, pos in enumerate(eval_positions):
                        original_token = input_ids[0, pos].item()
                        token_logits = logits[idx, pos]
                        log_probs = torch.log_softmax(token_logits, dim=0)
                        total_score += log_probs[original_token].item()
    
                    return total_score, len(eval_positions)
                
                # ===== ステップ4: 行別PLLを先に計算し、キャッシュ =====
                alpha = 0.6
                beta = 0.4
                temperature = 0.2
                
                pll_cache = {}
                per_entry_rescored = {}
                
                for entry_idx, entry in enumerate(entries):
                    clean_processed = entry['processed']
                    
                    # 置換前に全MASK位置を取得
                    mask_positions = []
                    start_search = 0
                    while True:
                        pos = clean_processed.find('[MASK]', start_search)
                        if pos == -1:
                            break
                        mask_positions.append(pos)
                        start_search = pos + len('[MASK]')
                    
                    entry_rescored = []
                    
                    for token, data in candidate_pool_filtered.items():
                        avg_prob = data['prob_sum'] / data['count']
                        
                        filled = clean_processed.replace('[MASK]', token)
                        
                        # 各MASK位置のウィンドウPLLを合算
                        delta = len('[MASK]') - len(token)
                        total_pll = 0.0
                        total_eval = 0
                        
                        for midx, mpos in enumerate(mask_positions):
                            adjusted_pos = mpos - midx * delta
                            adjusted_pos = max(0, min(adjusted_pos, len(filled) - 1))
                            pll_score_m, eval_count_m = evaluate_window_pll(
                                eval_model, filled, adjusted_pos
                            )
                            total_pll += pll_score_m
                            total_eval += eval_count_m
                        
                        pll_score = total_pll
                        eval_count = total_eval
                        normalized_pll = pll_score / max(eval_count, 1)
                        
                        if token not in pll_cache:
                            pll_cache[token] = {}
                        pll_cache[token][entry_idx] = {
                            'pll': pll_score,
                            'eval_count': eval_count,
                            'normalized_pll': normalized_pll
                        }
                        
                        entry_rescored.append({
                            'token': token,
                            'normalized_pll': normalized_pll,
                            'avg_prob': avg_prob,
                            'pll_score': pll_score,
                            'eval_count': eval_count
                        })
                    
                    if entry_rescored:
                        for item in entry_rescored:
                            log_avg_prob = math.log(item['avg_prob'] + 1e-12)
                            item['hybrid_log'] = alpha * item['normalized_pll'] + beta * log_avg_prob
                        
                        entry_rescored.sort(key=lambda x: -x['hybrid_log'])
                        max_log = entry_rescored[0]['hybrid_log']
                        exp_scores = [math.exp((x['hybrid_log'] - max_log) / temperature) for x in entry_rescored]
                        total = sum(exp_scores)
                        for i, item in enumerate(entry_rescored):
                            item['probability'] = exp_scores[i] / total if total > 0 else 0.0
                    
                    per_entry_rescored[entry_idx] = entry_rescored[:TOP_K]
                
                
                
                # ===== ステップ4A: 統一PLL（キャッシュから合算） =====
                unified_rescored = []
                num_entries = len(entries)
                
                for token, data in candidate_pool_filtered.items():
                    avg_prob = data['prob_sum'] / data['count']
                    
                    total_pll = 0.0
                    total_eval_count = 0
                    cached = pll_cache.get(token, {})
                    
                    for entry_idx in range(num_entries):
                        if entry_idx in cached:
                            total_pll += cached[entry_idx]['pll']
                            total_eval_count += cached[entry_idx]['eval_count']
                    
                    normalized_pll = total_pll / max(total_eval_count, 1)
                    
                    unified_rescored.append({
                        'token': token,
                        'normalized_pll': normalized_pll,
                        'avg_prob': avg_prob,
                        'pll_score': total_pll,
                        'eval_count': total_eval_count
                    })
                
                if unified_rescored:
                    for item in unified_rescored:
                        log_avg_prob = math.log(item['avg_prob'] + 1e-12)
                        item['hybrid_log'] = alpha * item['normalized_pll'] + beta * log_avg_prob
                    
                    unified_rescored.sort(key=lambda x: -x['hybrid_log'])
                    max_log = unified_rescored[0]['hybrid_log']
                    exp_scores = [math.exp((x['hybrid_log'] - max_log) / temperature) for x in unified_rescored]
                    total = sum(exp_scores)
                    for i, item in enumerate(unified_rescored):
                        item['probability'] = exp_scores[i] / total if total > 0 else 0.0
                
                unified_candidates = unified_rescored[:TOP_K]
                
                # ===== ステップ5: 全モデルに統一候補を適用 =====
                all_model_predictions = {}
                if not unified_candidates:
                    print("警告: 統一候補の生成に失敗。フォールバック処理")
                    for model_name, model in models.items():
                        predictions = model.predict_masks(entries[0]['processed'], use_full_context=USE_FULL_LINE_CONTEXT)
                        all_model_predictions[model_name] = predictions if predictions else []
                    display_model_names = model_names  # ★フォールバック時
                else:
                    all_model_predictions = {
                        "PLL": [{'position': 0, 'candidates': unified_candidates}]
                    }
                    display_model_names = ["PLL"]  # ★統一PLL時

                unified_ok = ("PLL" in all_model_predictions)
                ### 文ペア評価用モデル
                
            
            else:
                # 通常処理（単一MASK）
                unified_ok = False
                per_entry_rescored = {}
                display_model_names = model_names
                
                combined_processed = '\n'.join(entry['processed'] for entry in entries)
                all_model_predictions = {}
                for model_name, model in models.items():
                    all_model_predictions[model_name] = predict_masks_per_window(model, combined_processed, use_full_context=USE_FULL_LINE_CONTEXT)
                    
            # ★修正: 全MASKについて表示（mask_idxごと）
            pred_lengths = [len(preds) for preds in all_model_predictions.values() if len(preds) > 0]
            num_masks = max(pred_lengths) if pred_lengths else 0
            is_unified = unified_ok and (len(entries) >= 2 or total_masks >= 2)  # ★先に定義

            if is_unified:
                model_abbr = ["統一PLL"]
            else:
                model_abbr = [MODEL_ABBREVIATIONS.get(name, name) for name in display_model_names]
            
            for mask_idx in range(num_masks):
                if is_unified and len(entries) >= 2 and per_entry_rescored:
                    display_labels = ["統一PLL"] + [f"行{entry['line_num']}" for entry in entries]
                    predictions_for_display = []
                    
                    preds = all_model_predictions["PLL"]
                    if mask_idx < len(preds):
                        candidates = preds[mask_idx].get('candidates', [])
                        predictions_for_display.append([
                            (c['token'], c['probability'])
                            for c in candidates[:TOP_K]
                        ])
                    else:
                        predictions_for_display.append([])
                    
                    for entry_idx in range(len(entries)):
                        candidates = per_entry_rescored.get(entry_idx, [])
                        predictions_for_display.append([
                            (c['token'], c['probability'])
                            for c in candidates[:TOP_K]
                        ])
                
                elif is_unified:
                    display_labels = ["統一PLL"]
                    predictions_for_display = []
                    preds = all_model_predictions["PLL"]
                    if mask_idx < len(preds):
                        candidates = preds[mask_idx].get('candidates', [])
                        predictions_for_display.append([
                            (c['token'], c['probability'])
                            for c in candidates[:TOP_K]
                        ])
                
                else:
                    display_labels = model_abbr
                    predictions_for_display = []
                    for model_name in model_names:
                        if model_name in all_model_predictions:
                            preds = all_model_predictions[model_name]
                            if mask_idx < len(preds):
                                candidates = preds[mask_idx].get('candidates', [])
                                predictions_for_display.append([
                                    (c['token'], c['probability'])
                                    for c in candidates[:TOP_K]
                                ])
                            else:
                                predictions_for_display.append([])

                # 画面表示
                display_comparison_results(
                    predictions_for_display,
                    display_labels,
                    combined_original,
                    line_num=line_nums[0],
                    process_count=process_count,
                    show_prob=True,
                    decimals=PROB_DECIMALS_DISPLAY,
                    mask_type=mask_type,
                    group_lines=group_lines,
                    mask_index=mask_idx,
                    show_header=(mask_idx == 0),
                    f=None
                )
                    
                # ファイル出力
                with open(output_file, 'a', encoding='utf-8') as f:
                    display_comparison_results(
                        predictions_for_display,
                        display_labels,
                        combined_original,
                        line_num=line_nums[0],
                        process_count=process_count,
                        show_prob=True,
                        decimals=PROB_DECIMALS_FILE,
                        mask_type=mask_type,
                        group_lines=group_lines,
                        mask_index=mask_idx,
                        show_header=(mask_idx == 0),
                        f=f
                    )

            # アンサンブル計算・表示・保存（3方式）— 統一PLL時はスキップ
            if not is_unified:
                ensemble_by_method = {}
                for method in ENSEMBLE_METHODS:
                    if method == "rank":
                        ensemble_results = calculate_ensemble_scores_rank(all_model_predictions, model_names)
                    elif method == "probability":
                        ensemble_results = calculate_ensemble_scores_probability(all_model_predictions, model_names)
                    elif method == "hybrid":
                        ensemble_results = calculate_ensemble_scores_hybrid(all_model_predictions, model_names)
                    
                    ensemble_by_method[method] = ensemble_results
                    
                    display_ensemble_results(
                        ensemble_results=ensemble_results,
                        line_num=line_nums[0],
                        process_count=process_count,
                        original_text=combined_original,
                        mask_type=mask_type,
                        group_lines=group_lines,
                        method=method
                    )
                    
                    output_file_method = str(output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix))
                    save_ensemble_results(
                        output_file=output_file_method,
                        ensemble_results=ensemble_results,
                        line_num=line_nums[0],
                        process_count=process_count,
                        original_text=combined_original,
                        mask_type=mask_type,
                        group_lines=group_lines,
                        method=method
                    )
                
                # 3方式横並び出力
                display_ensemble_comparison(
                    ensemble_by_method, process_count,
                    mask_type=mask_type, group_lines=group_lines,
                    original_text=combined_original,
                    mask_count=num_masks,
                    f=None
                )
                ensemble_compare_file = str(output_path.with_name(output_path.stem + "_ensemble" + output_path.suffix))
                with open(ensemble_compare_file, 'a', encoding='utf-8') as ef:
                    display_ensemble_comparison(
                        ensemble_by_method, process_count,
                        mask_type=mask_type, group_lines=group_lines,
                        original_text=combined_original,
                        mask_count=num_masks,
                        f=ef
                    )

            all_results.append((line_nums[0], process_count, combined_original, all_model_predictions, model_names))

if __name__ == "__main__":
    main()