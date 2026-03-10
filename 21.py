import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import re
import sys
from pathlib import Path
import unicodedata
import math
import itertools

# ==================== 設定エリア ====================
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

MODEL_ABBREVIATIONS = {
    "hfl_chinese-macbert-base": "M-b",
    "hfl_chinese-macbert-large": "M-l",
    "hfl_chinese-bert-wwm-ext": "B-w",
    "hfl_chinese-roberta-wwm-ext": "R-w",
    "hfl_chinese-electra-180g-base-generator": "E-b",
    "hfl_chinese-electra-180g-large-generator": "E-l",
### "ckiplab_bert-base-chinese": "C-b",
    "ethanyt_guwenbert-base": "G-b",
    "bert-base-chinese": "B-b"
}

TOP_K = 10
USE_FULL_LINE_CONTEXT = True
SHOW_PROBABILITY = True
PROB_DECIMALS_DISPLAY = 4
PROB_DECIMALS_FILE = 4
MAX_TOKENS = 512
CANDIDATE_POOL_TOP_K = 80 # 60
PLL_CONTEXT_WINDOW_MIN = 30
PLL_CONTEXT_WINDOW_MAX = 80

# ==================== アンサンブルスコアリング設定 ====================
MIN_PROBABILITY_THRESHOLD = 0.015
MAX_RANK_TO_CONSIDER = 10
MIN_SUPPORT = 2
RANK_SCORE_METHOD = 'exponential'
ENSEMBLE_TOP_N = 10
ENSEMBLE_METHODS = ["rank", "probability", "hybrid"]

# ==================== 連続MASKの組み合わせ探索設定 ====================
# evaluate_consecutive_masks で使用
CONSECUTIVE_TOP_N = 10            # 各MASK位置の上位N候補
CONSECUTIVE_LIMIT_3MASK = 10      # 3連続MASKの1位置あたり候補数上限 (10^3=1000通り)
CONSECUTIVE_LIMIT_4MASK = 5       # 4連続MASKの1位置あたり候補数上限 (5^4=625通り)
CONSECUTIVE_COMBO_ALPHA = 0.6     # 組み合わせスコア重み（文脈考慮スコア）
CONSECUTIVE_COMBO_BETA  = 0.4     # 組み合わせスコア重み（独立スコア）

# ==================== PLL（文ペア）評価設定 ====================
# カスタムMASK処理の組み合わせ評価で使用
PLL_HYBRID_ALPHA = 0.6            # PLL正規化スコアの重み
PLL_HYBRID_BETA  = 0.4            # log確率スコアの重み
PLL_TEMPERATURE  = 0.6            # ソフトマックス温度（低い→上位集中 / 高い→分散） 0.7
PLL_LOG_PROB_EPS = 1e-12          # log計算のゼロ除算防止用ε

# ==================== モデル内部設定 ====================
OVERGEN_FACTOR_PREDICT  = 20      # predict_masks: top_k × この倍率で過剰生成してフィルタリング
OVERGEN_FACTOR_WINDOW   = 10      # predict_masks_per_window: 同上
MLM_SCORE_MAX_BATCH     = 64      # evaluate_mlm_score_fast: バッチサイズ上限
# =====================================================================


def display_ensemble_comparison(ensemble_by_method, process_count,
                                mask_type=None, group_lines=None,
                                original_text=None,
                                mask_count=1,
                                f=None):
    COL_WIDTH = 18
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

        method_results = []
        for method in methods:
            results = ensemble_by_method.get(method, [])
            if mask_idx < len(results):
                method_results.append(results[mask_idx])
            else:
                method_results.append([])

        max_rank = max((len(r) for r in method_results), default=0)

        if f is None:
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


def build_ensemble_output(ensemble_results, line_num, process_count, original_text,
                          mask_type=None, group_lines=None, method=None):
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


def display_ensemble_results(**kwargs):
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")
    output = build_ensemble_output(
        ensemble_results, kwargs["line_num"], kwargs["process_count"],
        kwargs["original_text"], mask_type=mask_type, group_lines=group_lines, method=method
    )
    print(output)


def save_ensemble_results(output_file, **kwargs):
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")
    output = build_ensemble_output(
        ensemble_results, kwargs["line_num"], kwargs["process_count"],
        kwargs["original_text"], mask_type=mask_type, group_lines=group_lines, method=method
    )
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(output + "\n")


def display_comparison_results(all_model_predictions, model_names, line_text,
                               line_num=None, process_count=None,
                               show_prob=True, decimals=4,
                               mask_type=None, group_lines=None,
                               mask_index=None, show_header=True, f=None):
    COL_WIDTH = 14
    
    def pad_to_width(s, width):
        w = 0
        for ch in s:
            if unicodedata.east_asian_width(ch) in ('F', 'W', 'A'):
                w += 2
            else:
                w += 1
        return s + " " * max(0, width - w)

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
        sentences = re.split(r'([。！？!?])', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]
    
    ALLOWED_PUNCT = {
        "，", "。", "！", "？", "、", "：", "；",
        "\u201c", "\u201d", "\u2018", "\u2019",
        "「", "」", "『", "』", "（", "）", "《", "》",
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
        if ch in ChineseMaskCompletion.ALLOWED_PUNCT:
            return True
        cat = unicodedata.category(ch)
        if cat.startswith("S"):
            return False
        if not (
            "\u4e00" <= ch <= "\u9fff" or
            "\u3400" <= ch <= "\u4dbf" or
            "\u20000" <= ch <= "\u2a6df" or
            "\u2a700" <= ch <= "\u2b73f"
        ):
            return False
        return True

    def evaluate_mlm_score_fast(self, text, skip_positions=None):
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
        all_logits = []
        for batch_start in range(0, batch_size, MLM_SCORE_MAX_BATCH):
            batch_end = min(batch_start + MLM_SCORE_MAX_BATCH, batch_size)
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
        if not use_full_context:
            sentences = self.split_sentences(text)
            mask_sentences = [s for s in sentences if '[MASK]' in s]
            if mask_sentences:
                return ''.join(mask_sentences)
            return text
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= MAX_TOKENS:
            return text
        sentences = self.split_sentences(text)
        mask_indices = [i for i, s in enumerate(sentences) if '[MASK]' in s]
        if not mask_indices:
            decoded = self.tokenizer.decode(tokens[:MAX_TOKENS], skip_special_tokens=True)
            return decoded
        first_mask = mask_indices[0]
        last_mask = mask_indices[-1]
        core = list(range(first_mask, last_mask + 1))
        context_indices = list(core)
        core_text = ''.join(sentences[i] for i in context_indices)
        core_tokens = len(self.tokenizer.encode(core_text, add_special_tokens=True))
        if core_tokens >= MAX_TOKENS:
            encoded = self.tokenizer.encode(core_text, add_special_tokens=True,
                                             truncation=True, max_length=MAX_TOKENS)
            return self.tokenizer.decode(encoded, skip_special_tokens=True)
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
            OVERGEN_K = min(self.top_k * OVERGEN_FACTOR_PREDICT, vocab_size)
            top_probs, top_indices = torch.topk(probs, OVERGEN_K)
            candidates = []
            for prob, idx in zip(top_probs, top_indices):
                token = self.tokenizer.convert_ids_to_tokens(int(idx))
                if not self.is_valid_chinese_token(token):
                    continue
                candidates.append({'token': token, 'probability': prob.item()})
                if len(candidates) >= self.top_k:
                    break
            results.append({'position': mask_idx, 'candidates': candidates})
        return results


def predict_masks_per_window(model, text, use_full_context=True):
    text = model.get_context_window(text, use_full_context)
    inputs = model.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    input_ids = inputs["input_ids"][0]
    mask_token_id = model.tokenizer.mask_token_id
    mask_positions_in_tokens = (input_ids == mask_token_id).nonzero(as_tuple=True)[0].tolist()
    if not mask_positions_in_tokens:
        return []
    full_tokens = input_ids.tolist()
    total_span = mask_positions_in_tokens[-1] - mask_positions_in_tokens[0] + 1
    if total_span + 2 <= MAX_TOKENS:
            return model.predict_masks(text, use_full_context=False)
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
        input_ids_t = torch.tensor(
            [[model.tokenizer.cls_token_id] + window_tokens + [model.tokenizer.sep_token_id]]
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids_t)
        with torch.no_grad():
            outputs = model.model(input_ids=input_ids_t, attention_mask=attention_mask)
            logits = outputs.logits
        mask_logits = logits[0, local_mask_pos + 1]
        probs = torch.softmax(mask_logits, dim=0)
        vocab_size = len(probs)
        OVERGEN_K = min(model.top_k * OVERGEN_FACTOR_WINDOW, vocab_size)
        top_probs, top_indices = torch.topk(probs, OVERGEN_K)
        candidates = []
        for prob, idx in zip(top_probs, top_indices):
            token = model.tokenizer.convert_ids_to_tokens(int(idx))
            if model.is_valid_chinese_token(token):
                candidates.append({'token': token, 'probability': prob.item()})
                if len(candidates) >= model.top_k:
                    break
        results.append({'position': mask_idx, 'candidates': candidates})
    return results


def evaluate_consecutive_masks(model, text, top_n=CONSECUTIVE_TOP_N, use_full_context=True): 
    context_text = model.get_context_window(text, use_full_context)
    inputs = model.tokenizer(context_text, return_tensors="pt",
                             truncation=True, max_length=MAX_TOKENS).to(model.device)
    mask_positions = (inputs.input_ids == model.tokenizer.mask_token_id).nonzero(as_tuple=True)[1].tolist()
    if len(mask_positions) < 2:
        return None
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
                    candidates.append({'tokens': [token], 'score': 0.0, 'probability': prob.item()})
                    if len(candidates) >= top_n:
                        break
            group_results.append(candidates)
            continue
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
        if len(group) == 2:
            combos = [(c1, c2) for c1 in position_candidates[0] for c2 in position_candidates[1]]
        elif len(group) == 3:
            limit = min(top_n, CONSECUTIVE_LIMIT_3MASK)
            combos = [(c1, c2, c3)
                      for c1 in position_candidates[0][:limit]
                      for c2 in position_candidates[1][:limit]
                      for c3 in position_candidates[2][:limit]]
        elif len(group) <= 4:
            limit = CONSECUTIVE_LIMIT_4MASK
            trimmed = [pc[:limit] for pc in position_candidates]
            combos = list(itertools.product(*trimmed))
        else:
            print(f"  警告: 連続{len(group)}MASKは計算コストが高いためスキップ")
            group_results.append([])
            continue
        scored_combos = []
        for combo in combos:
            test_ids = inputs.input_ids.clone()
            for i, (token, prob, token_id) in enumerate(combo):
                test_ids[0, group[i]] = token_id
            with torch.no_grad():
                test_outputs = model.model(test_ids)
                test_logits = test_outputs.logits
            combo_score = 0.0
            for i, (token, prob, token_id) in enumerate(combo):
                pos = group[i]
                pos_probs = torch.softmax(test_logits[0, pos], dim=0)
                combo_score += math.log(pos_probs[token_id].item() + 1e-12)
            indep_score = sum(math.log(c[1] + 1e-12) for c in combo)
            final_score = CONSECUTIVE_COMBO_ALPHA * combo_score + CONSECUTIVE_COMBO_BETA * indep_score
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
            token: data for token, data in token_scores.items()
            if len(data["models"]) >= MIN_SUPPORT
        }
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1]["score"], x[0]))
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({"token": token, "score": data["score"],
                                 "support": len(data["models"]), "details": data["models"]})
        ensemble_results.append(mask_result)
    return ensemble_results


def calculate_ensemble_scores_probability(all_model_predictions, model_names):
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
            token: data for token, data in token_scores.items()
            if len(data["models"]) >= MIN_SUPPORT
        }
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1]["score"], x[0]))
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({"token": token, "score": data["score"],
                                 "support": len(data["models"]), "details": data["models"]})
        ensemble_results.append(mask_result)
    return ensemble_results


def calculate_ensemble_scores_hybrid(all_model_predictions, model_names):
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
            token: data for token, data in token_scores.items()
            if len(data["models"]) >= MIN_SUPPORT
        }
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1]["score"], x[0]))
        mask_result = []
        for token, data in sorted_tokens[:ENSEMBLE_TOP_N]:
            mask_result.append({"token": token, "score": data["score"],
                                 "support": len(data["models"]), "details": data["models"]})
        ensemble_results.append(mask_result)
    return ensemble_results


def main():
    if len(sys.argv) < 2:
        print("使い方: python script.py <入力テキストファイル>")
        print("例: python script.py input.txt")
        return
    
    input_file = sys.argv[1]
    input_path = Path(input_file)
    output_path = input_path.with_name(input_path.stem + "_out" + input_path.suffix)
    output_file = str(output_path)
    
    if output_path.exists():
        output_path.unlink()
    for method in ENSEMBLE_METHODS:
        method_path = output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix)
        if method_path.exists():
            method_path.unlink()
    ensemble_compare_path = output_path.with_name(output_path.stem + "_ensemble" + output_path.suffix)
    if ensemble_compare_path.exists():
        ensemble_compare_path.unlink()
        
    models = {}
    for model_name in COMPARE_MODELS:
        model_path = f"models/{model_name}"
        if not Path(model_path).exists():
            print(f"警告: モデルが見つかりません: {model_path} (スキップします)")
            continue
        models[model_name] = ChineseMaskCompletion(model_path, top_k=TOP_K)
    
    if len(models) < MIN_SUPPORT:
        print(f"警告: ロード済みモデル数({len(models)})がMIN_SUPPORT({MIN_SUPPORT})未満です。")
    if not models:
        print("エラー: 使用可能なモデルがありません")
        return
    
    model_names = list(models.keys())
    
    if not Path(input_file).exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
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
                processed_line = re.sub(rf'\[MASK_{mask_type}\]', '##TARGET_MASK##', original_line)
                processed_line = re.sub(r'\[MASK_[A-Z0-9_]+\]', '', processed_line)
                processed_line = processed_line.replace('[MASK]', '')
                processed_line = processed_line.replace('##TARGET_MASK##', '[MASK]')
                if mask_type not in mask_groups:
                    mask_groups[mask_type] = []
                mask_groups[mask_type].append({
                    'line_num': line_num, 'original': original_line, 'processed': processed_line
                })
        if has_normal_mask:
            processed_line = re.sub(custom_mask_pattern, '', original_line)
            if 'MASK' not in mask_groups:
                mask_groups['MASK'] = []
            mask_groups['MASK'].append({
                'line_num': line_num, 'original': original_line, 'processed': processed_line
            })
    
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
                
                consecutive_results = None
                consec_text = None
                if combined_processed.count('[MASK]') >= 2:
                    consecutive_results = evaluate_consecutive_masks(
                        eval_model, combined_processed, top_n=CONSECUTIVE_TOP_N,
                        use_full_context=USE_FULL_LINE_CONTEXT
                    )
                    if consecutive_results:
                        consec_lines = [f"  ★連続MASK検出: 組み合わせ評価実行"]
                        for gi, group_cands in enumerate(consecutive_results):
                            if group_cands and len(group_cands[0].get('tokens', [])) >= 2:
                                consec_lines.append(f"    グループ{gi+1} (連続{len(group_cands[0]['tokens'])}字):")
                                for ri, cand in enumerate(group_cands[:5], 1):
                                    word = ''.join(cand['tokens'])
                                    consec_lines.append(f"      {ri}位: {word} ({cand['probability']:.4f})")
                        consec_text = "\n".join(consec_lines)

                all_model_predictions = {}
                for model_name, model in models.items():
                    predictions = predict_masks_per_window(
                        model, combined_processed, use_full_context=USE_FULL_LINE_CONTEXT
                    )
                    all_model_predictions[model_name] = predictions
                
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
                                    (c['token'], c['probability']) for c in candidates[:TOP_K]
                                ])
                            else:
                                predictions_for_display.append([])
                    display_comparison_results(
                        predictions_for_display, model_abbr, combined_original,
                        line_num=entry['line_num'], process_count=process_count,
                        show_prob=True, decimals=PROB_DECIMALS_DISPLAY,
                        mask_type=mask_type, group_lines=group_lines,
                        mask_index=mask_idx, show_header=(mask_idx == 0), f=None
                    )
                    with open(output_file, 'a', encoding='utf-8') as f:
                        display_comparison_results(
                            predictions_for_display, model_abbr, combined_original,
                            line_num=entry['line_num'], process_count=process_count,
                            show_prob=True, decimals=PROB_DECIMALS_FILE,
                            mask_type=mask_type, group_lines=group_lines,
                            mask_index=mask_idx, show_header=(mask_idx == 0), f=f
                        )
                
                if consec_text:
                    print(consec_text)
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(consec_text + "\n")
                
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
                        ensemble_results=ensemble_results, line_num=entry['line_num'],
                        process_count=process_count, original_text=entry['original'],
                        mask_type=mask_type, group_lines=group_lines, method=method
                    )
                    output_file_method = str(output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix))
                    save_ensemble_results(
                        output_file=output_file_method, ensemble_results=ensemble_results,
                        line_num=entry['line_num'], process_count=process_count,
                        original_text=entry['original'], mask_type=mask_type,
                        group_lines=group_lines, method=method
                    )
                
                display_ensemble_comparison(
                    ensemble_by_method, process_count,
                    mask_type=mask_type, group_lines=group_lines,
                    original_text=entry['original'], mask_count=num_masks, f=None
                )
                ensemble_compare_file = str(output_path.with_name(output_path.stem + "_ensemble" + output_path.suffix))
                with open(ensemble_compare_file, 'a', encoding='utf-8') as ef:
                    display_ensemble_comparison(
                        ensemble_by_method, process_count,
                        mask_type=mask_type, group_lines=group_lines,
                        original_text=entry['original'], mask_count=num_masks, f=ef
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

            if len(entries) >= 2 and len(set(mask_counts)) > 1:
                mask_count_info = [f'行{e["line_num"]}:{c}個' for e, c in zip(entries, mask_counts)]
                print(f"警告: MASK_{mask_type} グループ内でMASK数が一致していません: {mask_count_info}")

            if len(entries) >= 2 or total_masks >= 2:
                # ===== ステップ1: 全モデルから候補プールを収集（MASK位置ごと） =====
                all_mask_candidates_by_position = {}

                for entry in entries:
                    clean_processed = entry['processed']
                    for model_name, model in models.items():
                        preds = model.predict_masks(clean_processed, use_full_context=USE_FULL_LINE_CONTEXT)
                        if not preds:
                            continue
                        for mask_idx, pred in enumerate(preds):
                            if mask_idx not in all_mask_candidates_by_position:
                                all_mask_candidates_by_position[mask_idx] = {}
                            for cand in pred['candidates'][:CANDIDATE_POOL_TOP_K]:
                                token = cand['token']
                                prob = cand['probability']
                                if token not in all_mask_candidates_by_position[mask_idx]:
                                    all_mask_candidates_by_position[mask_idx][token] = {
                                        'prob_sum': 0.0, 'count': 0, 'details': []
                                    }
                                all_mask_candidates_by_position[mask_idx][token]['prob_sum'] += prob
                                all_mask_candidates_by_position[mask_idx][token]['count'] += 1
                                all_mask_candidates_by_position[mask_idx][token]['details'].append((model_name, prob))
                
                # ===== ステップ2: 各MASK位置でフィルタリング & Top-k選択 =====
                candidate_pool_filtered_by_position = {}
                for mask_idx, candidates_dict in all_mask_candidates_by_position.items():
                    sorted_candidates = sorted(
                        candidates_dict.items(),
                        key=lambda x: x[1]['prob_sum'] / x[1]['count'],
                        reverse=True
                    )
                    filtered = {}
                    for token, data in sorted_candidates[:CANDIDATE_POOL_TOP_K]:
                        if ChineseMaskCompletion.is_valid_chinese_token(token):
                            filtered[token] = data
                    if len(filtered) < max(5, CANDIDATE_POOL_TOP_K // 4):  # ★Fix4: 固定値5→動的閾値
                        for token, data in sorted_candidates:
                            if token in filtered:
                                continue
                            if ChineseMaskCompletion.is_valid_chinese_token(token):
                                filtered[token] = data
                                if len(filtered) >= CANDIDATE_POOL_TOP_K:
                                    break
                    candidate_pool_filtered_by_position[mask_idx] = filtered
                
                # ===== ウィンドウPLL評価関数 =====
                def evaluate_window_pll(model, filled_text, mask_char_pos, window_size=None):
                    if not getattr(model.tokenizer, "is_fast", False):
                        raise ValueError("evaluate_window_pll requires fast tokenizer")
                    inputs = model.tokenizer(filled_text, return_tensors="pt",
                                             truncation=True, max_length=MAX_TOKENS,
                                             return_offsets_mapping=True)
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
                    mask_token_pos = None
                    if offsets is not None:
                        for tidx, (start, end) in enumerate(offsets):
                            if start <= mask_char_pos < end:
                                mask_token_pos = tidx
                                break
                    if mask_token_pos is None:
                        print(f"警告: 文字位置{mask_char_pos}のトークン位置特定に失敗")
                        return 0.0, 1
                    if window_size is None:
                        adaptive_window = min(
                            PLL_CONTEXT_WINDOW_MAX,
                            max(PLL_CONTEXT_WINDOW_MIN, seq_len // 2)
                        )
                    else:
                        adaptive_window = window_size
                    if seq_len <= adaptive_window * 2 + 2:
                        skip_positions = {mask_token_pos}
                        score = model.evaluate_mlm_score_fast(filled_text, skip_positions=skip_positions)
                        eval_count = sum(
                            1 for i in range(seq_len)
                            if input_ids[0, i].item() not in special_ids and i != mask_token_pos
                        )
                        return score, max(eval_count, 1)
                    start = max(1, mask_token_pos - adaptive_window)
                    end = min(seq_len - 1, mask_token_pos + adaptive_window + 1)
                    eval_positions = []
                    for i in range(start, end):
                        token_id = input_ids[0, i].item()
                        if token_id in special_ids:
                            continue
                        if i == mask_token_pos:
                            continue
                        eval_positions.append(i)
                    if not eval_positions:
                        return 0.0, 1
                    MAX_BATCH = MLM_SCORE_MAX_BATCH
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
                
                # ===== ステップ4: 組み合わせ候補を生成してPLL評価 =====
                per_entry_rescored = {}

                mask_candidate_lists = []
                # ★Fix1(最重要): mask_idxが不連続の場合でも正しく最大値を算出
                num_masks = (max(all_mask_candidates_by_position.keys()) + 1
                             if all_mask_candidates_by_position else 0)

                if num_masks == 0:
                    print("警告: 候補プールが空です。フォールバック処理")
                    all_model_predictions = {}
                    for model_name, model in models.items():
                        all_model_predictions[model_name] = predict_masks_per_window(
                            model, combined_processed, use_full_context=USE_FULL_LINE_CONTEXT
                        )
                    display_model_names = model_names
                    unified_ok = False
                else:
                    for mask_idx in range(num_masks):
                        candidates = candidate_pool_filtered_by_position.get(mask_idx, {})
                        top_candidates = sorted(
                            candidates.items(),
                            key=lambda x: x[1]['prob_sum'] / x[1]['count'],
                            reverse=True
                        )[:10]
                        mask_candidate_lists.append([token for token, data in top_candidates])

                    if num_masks == 1:
                        combinations = [(token,) for token in mask_candidate_lists[0]]
                    elif num_masks == 2:
                        combinations = list(itertools.product(mask_candidate_lists[0], mask_candidate_lists[1]))
                    elif num_masks == 3:
                        limited_lists = [lst[:6] for lst in mask_candidate_lists]
                        combinations = list(itertools.product(*limited_lists))
                    else:
                        limited_lists = [lst[:3] for lst in mask_candidate_lists]
                        combinations = list(itertools.product(*limited_lists))

                    combination_scores = {}
                    pll_cache = {}  # ★Fix3: PLLキャッシュ追加

                    for combo in combinations:
                        combo_key = tuple(combo)

                        # ★注意点1対応: 幾何平均→log確率の平均に変更（PLLとの合成で数値安定）
                        log_prob_sum = 0.0
                        for mask_idx, token in enumerate(combo):
                            token_data = candidate_pool_filtered_by_position[mask_idx].get(token)
                            if token_data:
                                log_prob_sum += math.log(
                                    token_data['prob_sum'] / token_data['count'] + PLL_LOG_PROB_EPS
                                )
                        avg_log_prob = log_prob_sum / len(combo)  # 各位置のlog確率平均

                        combination_scores[combo_key] = {
                            'pll': 0.0,
                            'eval_count': 0,
                            'avg_log_prob': avg_log_prob,
                            'pll_per_entry': [0.0] * len(entries),
                            'eval_per_entry': [0] * len(entries)
                        }

                        for entry_idx, entry in enumerate(entries):
                            clean_processed = entry['processed']

                            # ★Fix2: re.finditerで安全にMASK位置を取得（文字列操作に依存しない）
                            mask_positions = [m.start() for m in re.finditer(r'\[MASK\]', clean_processed)]

                            filled = clean_processed
                            for mask_idx, token in enumerate(combo):
                                filled = filled.replace('[MASK]', token, 1)

                            total_pll = 0.0
                            total_eval = 0

                            for mask_idx, token in enumerate(combo):
                                if mask_idx >= len(mask_positions):
                                    continue
                                mpos = mask_positions[mask_idx]
                                adjusted_pos = mpos
                                for prev_idx in range(mask_idx):
                                    if prev_idx >= len(mask_positions):
                                        break
                                    delta = len('[MASK]') - len(combo[prev_idx])
                                    adjusted_pos -= delta
                                adjusted_pos = max(0, min(adjusted_pos, len(filled) - 1))

                                cache_key = (combo_key, entry_idx, mask_idx)
                                if cache_key not in pll_cache:
                                    pll_cache[cache_key] = evaluate_window_pll(
                                        eval_model, filled, adjusted_pos
                                    )
                                pll_score_m, eval_count_m = pll_cache[cache_key]

                                total_pll += pll_score_m
                                total_eval += eval_count_m

                            combination_scores[combo_key]['pll'] += total_pll
                            combination_scores[combo_key]['eval_count'] += total_eval
                            combination_scores[combo_key]['pll_per_entry'][entry_idx] += total_pll
                            combination_scores[combo_key]['eval_per_entry'][entry_idx] += total_eval

                    # 全行合計スコアでentry_rescoredを構築
                    entry_rescored = []
                    for combo, scores in combination_scores.items():
                        token = combo[0] if num_masks == 1 else ''.join(combo)
                        normalized_pll = scores['pll'] / max(scores['eval_count'], 1)
                        entry_rescored.append({
                            'token': token,
                            'normalized_pll': normalized_pll,
                            'avg_log_prob': scores['avg_log_prob'],
                            'pll_score': scores['pll'],
                            'eval_count': scores['eval_count']
                        })

                    if entry_rescored:
                        for item in entry_rescored:
                            item['hybrid_log'] = (PLL_HYBRID_ALPHA * item['normalized_pll']
                                                  + PLL_HYBRID_BETA  * item['avg_log_prob'])
                        entry_rescored.sort(key=lambda x: -x['hybrid_log'])
                        max_log = entry_rescored[0]['hybrid_log']
                        exp_scores = [math.exp((x['hybrid_log'] - max_log) / PLL_TEMPERATURE) for x in entry_rescored]
                        total = sum(exp_scores)
                        for i, item in enumerate(entry_rescored):
                            item['probability'] = exp_scores[i] / total if total > 0 else 0.0

                    # ★Fix1: per_entry_rescoredを行ごとに正しく再計算
                    if len(entries) >= 2:
                        for entry_idx in range(len(entries)):
                            entry_specific = []
                            for combo, scores in combination_scores.items():
                                token = combo[0] if num_masks == 1 else ''.join(combo)
                                pll = scores['pll_per_entry'][entry_idx]
                                eval_count = scores['eval_per_entry'][entry_idx]
                                normalized_pll = pll / max(eval_count, 1)
                                hybrid_log = (PLL_HYBRID_ALPHA * normalized_pll
                                              + PLL_HYBRID_BETA  * scores['avg_log_prob'])
                                entry_specific.append({
                                    'token': token,
                                    'normalized_pll': normalized_pll,
                                    'avg_log_prob': scores['avg_log_prob'],
                                    'hybrid_log': hybrid_log,
                                    'probability': 0.0
                                })
                            entry_specific.sort(key=lambda x: -x['hybrid_log'])
                            if entry_specific:
                                max_log_e = entry_specific[0]['hybrid_log']
                                exp_scores_e = [
                                    math.exp((x['hybrid_log'] - max_log_e) / PLL_TEMPERATURE)
                                    for x in entry_specific
                                ]
                                total_e = sum(exp_scores_e)
                                for i, item in enumerate(entry_specific):
                                    item['probability'] = exp_scores_e[i] / total_e if total_e > 0 else 0.0
                            per_entry_rescored[entry_idx] = entry_specific[:TOP_K]

                    # ===== ステップ4A: 統一候補を取得 =====
                    unified_candidates = entry_rescored[:TOP_K]

                    if not unified_candidates:
                        print("警告: 統一候補の生成に失敗。フォールバック処理")
                        for model_name, model in models.items():
                            predictions = predict_masks_per_window(
                                model, combined_processed, use_full_context=USE_FULL_LINE_CONTEXT
                            )
                            all_model_predictions[model_name] = predictions if predictions else []
                        display_model_names = model_names
                        unified_ok = False
                    else:
                        all_model_predictions = {
                            "PLL": [{'position': 0, 'candidates': unified_candidates}]
                        }
                        display_model_names = ["PLL"]
                        unified_ok = True
            
            else:
                # 通常処理（単一MASK）
                unified_ok = False
                per_entry_rescored = {}
                display_model_names = model_names
                combined_processed = '\n'.join(entry['processed'] for entry in entries)
                all_model_predictions = {}
                for model_name, model in models.items():
                    all_model_predictions[model_name] = predict_masks_per_window(
                        model, combined_processed, use_full_context=USE_FULL_LINE_CONTEXT
                    )
                    
            pred_lengths = [len(preds) for preds in all_model_predictions.values() if len(preds) > 0]
            num_masks = max(pred_lengths) if pred_lengths else 0
            is_unified = unified_ok and (len(entries) >= 2 or total_masks >= 2)

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
                            (c['token'], c['probability']) for c in candidates[:TOP_K]
                        ])
                    else:
                        predictions_for_display.append([])
                    for entry_idx in range(len(entries)):
                        candidates = per_entry_rescored.get(entry_idx, [])
                        predictions_for_display.append([
                            (c['token'], c['probability']) for c in candidates[:TOP_K]
                        ])
                elif is_unified:
                    display_labels = ["統一PLL"]
                    predictions_for_display = []
                    preds = all_model_predictions["PLL"]
                    if mask_idx < len(preds):
                        candidates = preds[mask_idx].get('candidates', [])
                        predictions_for_display.append([
                            (c['token'], c['probability']) for c in candidates[:TOP_K]
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
                                    (c['token'], c['probability']) for c in candidates[:TOP_K]
                                ])
                            else:
                                predictions_for_display.append([])

                display_comparison_results(
                    predictions_for_display, display_labels, combined_original,
                    line_num=line_nums[0], process_count=process_count,
                    show_prob=True, decimals=PROB_DECIMALS_DISPLAY,
                    mask_type=mask_type, group_lines=group_lines,
                    mask_index=mask_idx, show_header=(mask_idx == 0), f=None
                )
                with open(output_file, 'a', encoding='utf-8') as f:
                    display_comparison_results(
                        predictions_for_display, display_labels, combined_original,
                        line_num=line_nums[0], process_count=process_count,
                        show_prob=True, decimals=PROB_DECIMALS_FILE,
                        mask_type=mask_type, group_lines=group_lines,
                        mask_index=mask_idx, show_header=(mask_idx == 0), f=f
                    )

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
                        ensemble_results=ensemble_results, line_num=line_nums[0],
                        process_count=process_count, original_text=combined_original,
                        mask_type=mask_type, group_lines=group_lines, method=method
                    )
                    output_file_method = str(output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix))
                    save_ensemble_results(
                        output_file=output_file_method, ensemble_results=ensemble_results,
                        line_num=line_nums[0], process_count=process_count,
                        original_text=combined_original, mask_type=mask_type,
                        group_lines=group_lines, method=method
                    )
                display_ensemble_comparison(
                    ensemble_by_method, process_count,
                    mask_type=mask_type, group_lines=group_lines,
                    original_text=combined_original, mask_count=num_masks, f=None
                )
                ensemble_compare_file = str(output_path.with_name(output_path.stem + "_ensemble" + output_path.suffix))
                with open(ensemble_compare_file, 'a', encoding='utf-8') as ef:
                    display_ensemble_comparison(
                        ensemble_by_method, process_count,
                        mask_type=mask_type, group_lines=group_lines,
                        original_text=combined_original, mask_count=num_masks, f=ef
                    )

            all_results.append((line_nums[0], process_count, combined_original, all_model_predictions, model_names))

if __name__ == "__main__":
    main()
