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
    # KLUE系（CC BY-SA 4.0）
    "klue/roberta-base",
    "klue/roberta-large",
    "klue/roberta-small", 
    "klue/bert-base",

    # Beomi系
    "beomi/kcbert-base",
### "beomi/KcELECTRA-base",  # 除外（語彙の99.6%が非ハングル）
    "beomi/KcELECTRA-base-v2022",
    
    # ELECTRA系
    "monologg/koelectra-base-v3-discriminator",
    "tunib/electra-ko-base", 
    
    # その他
    "lassl/bert-ko-base",
    "jinmang2/kpfbert", 
    "monologg/distilkobert",
### "monologg/kobert"
### "kykim/bert-kor-base",
]

# モデル名の略称
MODEL_ABBREVIATIONS = {
    # KLUE系
    "klue/roberta-base": "KR-b",
    "klue/roberta-large": "KR-l",
    "klue/roberta-small": "KR-s",
    "klue/bert-base": "KB-b",

    # Beomi系
    "beomi/kcbert-base": "KC-b",
### "beomi/KcELECTRA-base": "KcE-b",
    "beomi/KcELECTRA-base-v2022": "KcE-22",
    
    # ELECTRA系
    "monologg/koelectra-base-v3-discriminator": "KE-v3",
    "tunib/electra-ko-base": "TE-b",
    
    # その他
    "lassl/bert-ko-base": "LS-b",
    "jinmang2/kpfbert": "KPF-b",
    "monologg/distilkobert": "DK-b",
### "kykim/bert-kor-base": "KK-b",
### "monologg/kobert": "KoB"
}

# 表示候補数（モデル別・アンサンブル共通）
# 推奨: 5〜20。増やすと低確率候補も見えるが画面が広がる
TOP_K = 10

# 文脈考慮モード（True: 行全体を考慮, False: MASKを含む1文のみ考慮）
# 推奨: True。長文で512トークンを超える場合は自動的にウィンドウ処理に切り替わる
USE_FULL_LINE_CONTEXT = True

# 確率表示（True: 表示, False: 非表示）
SHOW_PROBABILITY = True

# 確率の小数点以下桁数
# 推奨: 画面4、ファイル4。詳細比較が必要なら6に増やす
PROB_DECIMALS_DISPLAY = 4  # 画面表示用
PROB_DECIMALS_FILE = 4     # ファイル出力用

# モデルの最大トークン数
# 通常は512固定。モデルが128や256の場合はそれに合わせる
MAX_TOKENS = 512

# 字母補完モード（True: 有効, False: 無効）
# 入力にㄱ〜ㅎ（初声）や母音字母を含む場合に完成形ハングルへ変換してMASK化する
JAMO_MODE = True

# 温度パラメータ（predict_masks の確率分布調整）
# 1.0=標準分布, >1.0で分散（多様な候補を得たい場合）, <1.0で上位集中
# 推奨範囲: 0.8〜2.0。低すぎると上位1〜2候補に偏り、高すぎると確率が均等化される
TEMPERATURE = 1.2

# PLLスコアリング前の候補プールサイズ（モデルあたりの上位N件を収集）
# 推奨範囲: 30〜100。大きいほど多様な候補をPLLで評価できるが計算コスト増
# GPUメモリが少ない場合は30〜50に下げる
CANDIDATE_POOL_TOP_K = 80

# ==================== アンサンブルスコアリング設定 ====================
# アンサンブル対象から除外する確率閾値（この値未満を無視）
# 推奨範囲: 0.005〜0.05。低すぎるとノイズ候補が混入、高すぎると有効候補が消える
MIN_PROBABILITY_THRESHOLD = 0.015

# アンサンブルで評価する最大順位（各モデルの上位N位まで集計）
# 推奨範囲: 5〜20。MIN_SUPPORT と合わせて調整する
MAX_RANK_TO_CONSIDER = 10

# アンサンブルに必要な最小支持モデル数（この数以上のモデルが推薦した候補のみ残す）
# 推奨: モデル総数の1/3〜1/2程度。モデルを大幅に追加・削除した場合は要調整
MIN_SUPPORT = 2

# 順位スコアの計算方式
# 'exponential'（推奨）: 上位に指数的に高いスコア（1位と2位の差が大きい）
# 'logarithmic': 対数的に逓減（緩やかな差）
# 'linear': 線形に逓減
# 'inverse': 順位の逆数
RANK_SCORE_METHOD = 'exponential'

# アンサンブル結果の表示候補数
# 推奨: TOP_K と同じか大きい値。TOP_K より小さくすると表示が減る
ENSEMBLE_TOP_N = 10

# 使用するアンサンブル方式（不要な方式はコメントアウト可）
ENSEMBLE_METHODS = ["rank", "probability", "hybrid"]
# =====================================================================

# ==================== PLL（文ペア）評価設定 ====================
# PLL正規化スコアと log確率スコアの重み（合計1.0が目安）
# ALPHA大: 文脈適合性を重視 / BETA大: モデルの語彙頻度を重視
# 推奨: 0.6/0.4 〜 0.7/0.3。複数行にまたがる場合はALPHAを上げる
PLL_HYBRID_ALPHA = 0.6
PLL_HYBRID_BETA  = 0.4

# PLLスコアを確率に変換するソフトマックス温度
# 低い（0.3〜0.5）→ 上位候補に集中 / 高い（0.8〜1.5）→ 確率が分散
# 推奨: 0.5〜0.8
PLL_TEMPERATURE  = 0.6

# log計算のゼロ除算防止用ε（通常変更不要）
PLL_LOG_PROB_EPS = 1e-12

# PLLウィンドウサイズの適応範囲（シーケンス長に応じてMIN〜MAXの間で自動決定）
# MIN を下げると短文でも広い文脈を参照、MAX を上げると長文での計算コスト増
# 推奨: MIN=20〜40、MAX=60〜100。GPUメモリが少ない場合はMAXを下げる
PLL_CONTEXT_WINDOW_MIN = 30
PLL_CONTEXT_WINDOW_MAX = 80

# ==================== モデル内部設定 ====================
# PLLバッチ処理のサイズ上限（大きいほど高速だがGPUメモリを消費）
# 推奨: GPU 8GB以上→64、4GB→32、CPUのみ→16
MLM_SCORE_MAX_BATCH = 64
# =====================================================================


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


# 韓国語字母の定義
CHOSUNG = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNGSUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONGSUNG = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

# 韓国語で許可する句読点・記号
ALLOWED_PUNCT_KO = {
    ".", ",", "!", "?", ":", ";",
    "。", "，", "！", "？", "、", "：", "；",
    "\u201c", "\u201d",  # ""
    "\u2018", "\u2019",  # ''
    "「", "」", "『", "』",
    "（", "）", "《", "》",
    "(", ")",
    "—", "…", "·",
    "~", "-",
}

def get_chosung(char):
    """ハングル文字から初声を抽出"""
    if not ('가' <= char <= '힣'):
        return None
    code = ord(char) - ord('가')
    chosung_index = code // (21 * 28)
    return CHOSUNG[chosung_index]

def get_jungsung(char):
    """ハングル文字から中声を抽出"""
    if not ('가' <= char <= '힣'):
        return None
    code = ord(char) - ord('가')
    jungsung_index = (code % (21 * 28)) // 28
    return JUNGSUNG[jungsung_index]

def get_jongsung(char):
    """ハングル文字から終声を抽出"""
    if not ('가' <= char <= '힣'):
        return None
    code = ord(char) - ord('가')
    jongsung_index = code % 28
    if jongsung_index == 0:
        return None
    return JONGSUNG[jongsung_index - 1]


def match_jamo_pattern(word, jamo_pattern):
    """字母パターンにマッチするかチェック"""
    if not word or not jamo_pattern:
        return False
    
    first_jamo = jamo_pattern[0]
    
    # 完成形ハングル → prefix一致
    if '\uAC00' <= first_jamo <= '\uD7A3':
        return word.startswith(jamo_pattern)
    
    # 初声のみ → 初声一致
    if first_jamo in CHOSUNG:
        if word and '\uAC00' <= word[0] <= '\uD7A3':
            return get_chosung(word[0]) == first_jamo
        return False
    
    # その他 → prefix一致
    return word.startswith(jamo_pattern)


def replace_jamos_with_masks(line):
    """字母を完成形ハングルに変換してから[MASK]に置換し、位置情報を返す
    
    Returns:
        result: 変換後のテキスト
        jamo_info: [(jamo_index, jamo_pattern, position_in_result), ...]
            position_in_result: 結果文字列中での[MASK]の開始位置
    """
    jamo_info = []
    result = []
    
    i = 0
    while i < len(line):
        char = line[i]
        converted = None
        jamo_len = 0
        
        # カスタムマスクをスキップ
        if char == '[':
            match = re.match(r'\[MASK[^\]]*\]', line[i:])
            if match:
                result.append(match.group())
                i += len(match.group())
                continue
        
        # 初声+中声+終声（3文字）
        if (i + 2 < len(line) and 
            line[i] in CHOSUNG and 
            line[i + 1] in JUNGSUNG and 
            line[i + 2] in JONGSUNG):
            cho_idx = CHOSUNG.index(line[i])
            jung_idx = JUNGSUNG.index(line[i + 1])
            jong_idx = JONGSUNG.index(line[i + 2]) + 1
            converted = chr(0xAC00 + cho_idx * 588 + jung_idx * 28 + jong_idx)
            jamo_len = 3
        
        # 初声+中声（2文字）
        elif i + 1 < len(line) and line[i] in CHOSUNG and line[i + 1] in JUNGSUNG:
            cho_idx = CHOSUNG.index(line[i])
            jung_idx = JUNGSUNG.index(line[i + 1])
            converted = chr(0xAC00 + cho_idx * 588 + jung_idx * 28)
            jamo_len = 2
        
        # 初声のみ（1文字）
        elif char in CHOSUNG:
            converted = char
            jamo_len = 1
        
        if converted:
            # 直後にカスタムマスク [MASK_X] があるか確認
            next_pos = i + jamo_len
            next_is_custom_mask = bool(re.match(r'\[MASK_[A-Z0-9_]+\]', line[next_pos:]))
            if next_is_custom_mask:
                # 字母はそのまま残す（カスタムマスク側で紐付ける）
                result.append(line[i:i + jamo_len])
                i += jamo_len
            else:
                # 結果文字列中での現在位置を計算
                current_pos = sum(len(s) for s in result)
                mask_index = len(jamo_info)
                jamo_info.append((mask_index, converted, current_pos))
                result.append('[MASK]')
                i += jamo_len
        else:
            result.append(char)
            i += 1
    
    return ''.join(result), jamo_info


# ==================== アンサンブル表示共通化 ====================
def build_ensemble_output(ensemble_results,
                          line_num,
                          process_count,
                          original_text,
                          mask_type=None,
                          group_lines=None,
                          method=None,
                          jamo_map=None):

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
        if mask_type and mask_type not in ('MASK', 'MASK_JAMO'):
            lines.append(f"[MASK_{mask_type}]:")
        elif jamo_map and idx in jamo_map:
            lines.append(f"[字母]{idx+1}（{jamo_map[idx]}）:")
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
    """画面表示"""
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")
    jamo_map = kwargs.get("jamo_map")
    
    output = build_ensemble_output(
        ensemble_results,
        kwargs["line_num"],
        kwargs["process_count"],
        kwargs["original_text"],
        mask_type=mask_type,
        group_lines=group_lines,
        method=method,
        jamo_map=jamo_map
    )
    print(output)


def save_ensemble_results(output_file, **kwargs):
    """ファイル保存"""
    ensemble_results = kwargs["ensemble_results"]
    mask_type = kwargs.get("mask_type")
    group_lines = kwargs.get("group_lines")
    method = kwargs.get("method", "hybrid")
    jamo_map = kwargs.get("jamo_map")

    output = build_ensemble_output(
        ensemble_results,
        kwargs["line_num"],
        kwargs["process_count"],
        kwargs["original_text"],
        mask_type=mask_type,
        group_lines=group_lines,
        method=method,
        jamo_map=jamo_map
    )

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(output + "\n")





def _format_ensemble_cell(items, rank, method):
    if rank >= len(items):
        return ""
    item = items[rank]
    if isinstance(item, dict):
        token = item['token']
        score = item['score']
        if method == "rank":
            return f"{token} ({score:.2f}点)"
        elif method == "probability":
            return f"{token} ({score:.4f})"
        elif method == "hybrid":
            return f"{token} ({score:.3f}点)"
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        token, score = item
        return f"{token} ({score:.6f})"
    return ""


def display_and_save_ensemble_combined(output_file, results_by_method,
                                        line_num, process_count, original_text,
                                        mask_type=None, group_lines=None,
                                        jamo_map=None):
    """3方式横並びの画面表示とファイル保存"""
    header_lines = []
    header_lines.append("=" * 80)
    header_lines.append(f"{process_count}番目")

    if mask_type and group_lines:
        if mask_type == 'MASK':
            header_lines.append("通常MASKグループ:")
        else:
            header_lines.append(f"MASK_{mask_type}グループ:")
        for gline_num, gline_text in group_lines:
            header_lines.append(f"  行{gline_num}: {gline_text}")
    else:
        header_lines.append(original_text)
    header_lines.append("-" * 80)

    rank_results = results_by_method.get("rank", [])
    prob_results = results_by_method.get("probability", [])
    hybrid_results = results_by_method.get("hybrid", [])

    num_masks = max(len(rank_results), len(prob_results), len(hybrid_results))

    console_lines = header_lines.copy()
    file_lines = header_lines.copy()

    methods = ["rank", "probability", "hybrid"]

    for idx in range(num_masks):
        if mask_type and mask_type not in ('MASK', 'MASK_JAMO'):
            label = f"[MASK_{mask_type}]:"
        elif jamo_map and idx in jamo_map:
            label = f"[字母]{idx+1}（{jamo_map[idx]}）:"
        else:
            label = f"[MASK]{idx+1}:"

        console_lines.append(label)
        file_lines.append(label)

        h_console = "      " + "".join([f"{m:<22}" for m in methods])
        h_file = "      \t" + "\t".join(methods)
        console_lines.append(h_console)
        file_lines.append(h_file)

        r_items = rank_results[idx] if idx < len(rank_results) else []
        p_items = prob_results[idx] if idx < len(prob_results) else []
        h_items = hybrid_results[idx] if idx < len(hybrid_results) else []

        max_rank = max(len(r_items), len(p_items), len(h_items))
        max_rank = min(max_rank, ENSEMBLE_TOP_N)

        for rank in range(max_rank):
            r_cell = _format_ensemble_cell(r_items, rank, "rank")
            p_cell = _format_ensemble_cell(p_items, rank, "probability")
            h_cell = _format_ensemble_cell(h_items, rank, "hybrid")

            prefix = f"{rank+1:>2}位:  "
            console_lines.append(prefix + r_cell.ljust(22) + p_cell.ljust(22) + h_cell.ljust(22))
            file_lines.append(prefix + "\t".join([r_cell, p_cell, h_cell]))

        console_lines.append("")
        file_lines.append("")

    print("\n".join(console_lines))
    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n".join(file_lines) + "\n")


def display_comparison_results(all_model_predictions, model_names, line_text,
                               line_num=None, process_count=None,
                               show_prob=True, decimals=4,
                               mask_type=None, group_lines=None,
                               mask_index=None,
                               show_header=True,
                               f=None,
                               jamo_map=None):

    lines = []
    
    if show_header:
        lines.append("=" * 80)
        lines.append(f"{process_count}番目")
        lines.append("-" * 80)

        if mask_type and group_lines:
            if mask_type == 'MASK':
                lines.append("通常MASKグループ:")
            else:
                lines.append(f"MASK_{mask_type}グループ:")

            for gline_num, gline_text in group_lines:
                lines.append(f"  行{gline_num}: {gline_text}")

            lines.append("-" * 80)
        else:
            lines.append(line_text)
            lines.append("-" * 80)

    # MASKインデックス表示
    if mask_index is not None:
        if mask_type and mask_type not in ('MASK', 'MASK_JAMO'):
            lines.append(f"[MASK_{mask_type}]:")
        elif jamo_map and mask_index in jamo_map:
            lines.append(f"[字母]{mask_index+1}（{jamo_map[mask_index]}）:")
        else:
            lines.append(f"[MASK]{mask_index + 1}:")

    header_console = "      " + "".join([f"{name:<14}" for name in model_names])
    header_file = "      \t" + "\t".join(model_names)

    if not any(isinstance(p, list) and len(p) > 0 for p in all_model_predictions):
        lines_console = lines.copy()
        lines_console.append(header_console)
        lines_console.append("予測結果なし")
        lines_file = lines.copy()
        lines_file.append(header_file)
        lines_file.append("予測結果なし")
        print("\n".join(lines_console))
        if f is not None:
            f.write("\n".join(lines_file) + "\n")
        return

    max_rank = max((len(p) for p in all_model_predictions if isinstance(p, list)), default=0)

    rows_console = []
    rows_file = []
    for rank in range(min(max_rank, TOP_K)):
        row_console = f"{rank+1:>2}位:  "
        row_file_cells = [f"{rank+1:>2}位:"]

        for model_preds in all_model_predictions:
            if isinstance(model_preds, list) and rank < len(model_preds):
                item = model_preds[rank]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    token, prob = item
                    if show_prob:
                        cell = f"{token} ({prob:.{decimals}f})"
                    else:
                        cell = f"{token}"
                else:
                    cell = f"{item}"
            else:
                cell = ""
            row_console += cell.ljust(14)
            row_file_cells.append(cell)

        rows_console.append(row_console)
        rows_file.append("\t".join(row_file_cells))

    lines_console = lines.copy()
    lines_console.append(header_console)
    lines_console.extend(rows_console)

    lines_file = lines.copy()
    lines_file.append(header_file)
    lines_file.extend(rows_file)

    if f is None:
        print("\n" + "\n".join(lines_console))
    else:
        f.write("\n".join(lines_file) + "\n\n")

# ==================== モデルクラス ====================
class KoreanMaskCompletion:
    def __init__(self, model_path, top_k=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")

        kwargs = {}
        if "distilkobert" in model_path.lower():
            kwargs["trust_remote_code"] = True
    
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(model_path, **kwargs).to(self.device)
        self.model.eval()
        self.mask_token_id = self.tokenizer.mask_token_id
        self.top_k = top_k
        self.mask_token = self.tokenizer.mask_token
    
        # Token IDキャッシュ構築
        # len(tokenizer) を使用（vocab_size はモデルによって実語彙数と一致しない場合がある）
        vocab_size = len(self.tokenizer)
        tokens_list = self.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        self.id2token_cache = dict(enumerate(tokens_list))
        self.unk_token = self.tokenizer.unk_token
        
        # ハングルトークンIDセットの事前構築
        self.hangul_token_ids = set()
    
        for token_id in range(vocab_size):
            token = self.id2token_cache.get(token_id)
            if token is None:
                continue
        
            if token_id in self.tokenizer.all_special_ids:
                continue
        
            token_clean = token.lstrip('▁ĠĊ▃')
            if token_clean.startswith('##'):
                token_clean = token_clean[2:]
        
            if not token_clean:
                continue
        
            if re.fullmatch(r'[\uAC00-\uD7A3]+', token_clean):
                has_jamo = any(ord(c) in range(0x1100, 0x1200) or ord(c) in range(0x3130, 0x3190) for c in token_clean)
                if not has_jamo:
                    self.hangul_token_ids.add(token_id)
    
        # 句読点・記号トークンIDセットの事前構築
        self.punct_token_ids = set()
        for token_id in range(vocab_size):
            if token_id in self.tokenizer.all_special_ids:
                continue
            token = self.id2token_cache.get(token_id)
            if token is None:
                continue
            token_clean = token.lstrip('▁ĠĊ▃')
            if token_clean.startswith('##'):
                token_clean = token_clean[2:]
            if len(token_clean) == 1 and token_clean in ALLOWED_PUNCT_KO:
                self.punct_token_ids.add(token_id)

        # ハングル＋許可記号の統合セット
        self.valid_token_ids = self.hangul_token_ids | self.punct_token_ids

        # TensorにしてGPUに転送
        valid_ids_list = sorted(self.valid_token_ids)
        self.valid_ids_tensor = torch.tensor(valid_ids_list, dtype=torch.long, device=self.device)
        
        # ハングルのみのTensor（字母フィルタ用）
        hangul_ids_list = sorted(self.hangul_token_ids)
        self.hangul_ids_tensor = torch.tensor(hangul_ids_list, dtype=torch.long, device=self.device)
        
        print(f"Hangul tokens: {len(self.hangul_token_ids)}/{vocab_size} "
              f"({100*len(self.hangul_token_ids)/vocab_size:.1f}%)")
        print(f"Punct tokens: {len(self.punct_token_ids)}, "
              f"Valid total: {len(self.valid_token_ids)}")
    
        try:
            import inspect
            forward_params = inspect.signature(self.model.forward).parameters
            self._use_token_type_ids = 'token_type_ids' in forward_params
        except Exception:
            model_class = self.model.__class__.__name__.lower()
            self._use_token_type_ids = not any(
                x in model_class for x in ('distil', 'roberta', 'electra')
            )
            
    @staticmethod
    def get_display_width(text):
        """文字列の表示幅を計算"""
        width = 0
        for char in text:
            ea_width = unicodedata.east_asian_width(char)
            if ea_width in ('F', 'W'):
                width += 2
            else:
                width += 1
        return width
    
    def split_sentences(self, text):
        """テキストを文に分割"""
        sentences = re.split(r'([.!?。！？])', text)
        result = []
        for i in range(0, len(sentences)-1, 2):
            if i+1 < len(sentences):
                result.append(sentences[i] + sentences[i+1])
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            result.append(sentences[-1])
        return [s for s in result if s.strip()]

    @staticmethod
    def is_valid_korean_token(token: str) -> bool:
        """韓国語の有効なトークンかチェック（ハングル＋許可記号）"""
        if not token:
            return False
        
        if token in {"<UNK>", "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"}:
            return False
        
        if token.startswith("[") and token.endswith("]"):
            return False
        
        # プレフィックス除去（##はBERT系サブワード接頭辞なので除去してから判定）
        token_clean = token.lstrip('▁ĠĊ▃')
        if token_clean.startswith('##'):
            token_clean = token_clean[2:]

        if not token_clean:
            return False

        # 1文字の場合: 完成形ハングルまたは許可句読点
        if len(token_clean) == 1:
            ch = token_clean[0]
            if '\uAC00' <= ch <= '\uD7A3':
                return True
            if ch in ALLOWED_PUNCT_KO:
                return True
            return False

        # 複数文字の場合: 完成形ハングルのみ
        if re.fullmatch(r'[\uAC00-\uD7A3]+', token_clean):
            has_jamo = any(ord(c) in range(0x1100, 0x1200) or ord(c) in range(0x3130, 0x3190) for c in token_clean)
            if not has_jamo:
                return True

        return False

    def get_context_window(self, text, use_full_context=True):
        """文脈ウィンドウを取得（全MASK位置を含む、512トークン制約考慮）"""
        max_tokens = getattr(self.tokenizer, 'model_max_length', 512)
        if max_tokens > 100000:
            max_tokens = 512

        if not use_full_context:
            sentences = self.split_sentences(text)
            mask_sentences = [s for s in sentences if self.mask_token in s]
            if mask_sentences:
                return ''.join(mask_sentences)
            return text

        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= max_tokens:
            return text

        sentences = self.split_sentences(text)
        
        # 全MASKを含む文のインデックスを取得
        mask_indices = [i for i, s in enumerate(sentences) if self.mask_token in s]

        if not mask_indices:
            decoded = self.tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
            return decoded

        # 最初のMASK〜最後のMASKを含む範囲を必須コアにする
        first_mask = mask_indices[0]
        last_mask = mask_indices[-1]

        core = list(range(first_mask, last_mask + 1))
        context_indices = list(core)

        core_text = ''.join(sentences[i] for i in context_indices)
        core_tokens = len(self.tokenizer.encode(core_text, add_special_tokens=True))
        if core_tokens >= max_tokens:
            encoded = self.tokenizer.encode(core_text, add_special_tokens=True,
                                             truncation=True, max_length=max_tokens)
            return self.tokenizer.decode(encoded, skip_special_tokens=True)

        # コアから前後に拡張
        left = first_mask - 1
        right = last_mask + 1

        while left >= 0 or right < len(sentences):
            current_text = ''.join(sentences[i] for i in sorted(context_indices))
            current_tokens = len(self.tokenizer.encode(current_text, add_special_tokens=True))
            
            if current_tokens >= max_tokens:
                break
            
            added = False
            if left >= 0:
                test_indices = sorted(context_indices + [left])
                test_text = ''.join(sentences[i] for i in test_indices)
                if len(self.tokenizer.encode(test_text, add_special_tokens=True)) <= max_tokens:
                    context_indices.append(left)
                    left -= 1
                    added = True
                else:
                    left = -1
            
            if right < len(sentences):
                test_indices = sorted(context_indices + [right])
                test_text = ''.join(sentences[i] for i in test_indices)
                if len(self.tokenizer.encode(test_text, add_special_tokens=True)) <= max_tokens:
                    context_indices.append(right)
                    right += 1
                    added = True
                else:
                    right = len(sentences)
            
            if not added:
                break

        return ''.join(sentences[i] for i in sorted(context_indices))

    def predict_masks(self, text, jamo_info=None, use_full_context=True):
        """[MASK]トークンを予測"""
        text = text.replace("[MASK]", self.mask_token)  # モデル固有のmask_tokenに変換
        context_text = self.get_context_window(text, use_full_context)

        inputs = self.tokenizer(context_text, return_tensors="pt",
                                truncation=True, max_length=MAX_TOKENS)
        input_ids = inputs['input_ids'].to(self.device)

        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        token_type_ids = None
        if 'token_type_ids' in inputs:
            token_type_ids = inputs['token_type_ids'].to(self.device)

        mask_token_indices = (input_ids == self.mask_token_id).nonzero(as_tuple=True)[1]

        if len(mask_token_indices) == 0:
            return []

        if not self._use_token_type_ids:
            token_type_ids = None
    
        with torch.no_grad():
            if token_type_ids is not None:
                outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = self.model(input_ids, attention_mask=attention_mask)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs.logits

        # jamo_infoは2タプル(idx, pattern)または3タプル(idx, pattern, pos)
        jamo_map = {}
        if jamo_info:
            for ji in jamo_info:
                jamo_map[ji[0]] = ji[1]

        results = []
        for mask_pos_idx, token_idx in enumerate(mask_token_indices):
            mask_logits = logits[0, token_idx]
            jamo_pattern = jamo_map.get(mask_pos_idx)

            scaled_logits = mask_logits / TEMPERATURE

            if jamo_pattern:
                # 字母あり: ハングルトークンのみに絞る
                top_k = min(200, len(self.hangul_ids_tensor))
                hangul_logits = scaled_logits[self.hangul_ids_tensor]
                top_vals, top_indices = torch.topk(hangul_logits, top_k)
                top_ids = self.hangul_ids_tensor[top_indices]
            else:
                # 字母なし: ハングル＋許可記号
                top_k = min(100, len(self.valid_ids_tensor))
                valid_logits = scaled_logits[self.valid_ids_tensor]
                top_vals, top_indices = torch.topk(valid_logits, top_k)
                top_ids = self.valid_ids_tensor[top_indices]

            top_probs = torch.softmax(top_vals, dim=0)

            candidates = []
            for i, p in zip(top_ids, top_probs):
                token_id = i.item()
                
                if token_id in self.tokenizer.all_special_ids:
                    continue

                token = self.id2token_cache.get(token_id, self.unk_token)
                
                token_clean = token.lstrip('▁ĠĊ▃')
                if token_clean.startswith('##'):
                    token_clean = token_clean[2:]

                if not token_clean:
                    continue

                # 字母フィルタリング
                if jamo_pattern:
                    # 字母モードではハングルのみ
                    if not re.fullmatch(r'[\uAC00-\uD7A3]+', token_clean):
                        continue
                    if any(ord(c) in range(0x1100, 0x1200) or ord(c) in range(0x3130, 0x3190) for c in token_clean):
                        continue
                    if not match_jamo_pattern(token_clean, jamo_pattern):
                        continue
                else:
                    # 非字母モード: ハングルまたは許可句読点
                    if not self.is_valid_korean_token(token):
                        continue

                candidates.append({
                    "token": token_clean,
                    "probability": p.item()
                })

            # 確率正規化
            if candidates:
                total_prob = sum(c["probability"] for c in candidates)
                if total_prob > 0:
                    for c in candidates:
                        c["probability"] /= total_prob

            candidates = sorted(candidates, key=lambda x: x["probability"], reverse=True)
            candidates = candidates[:self.top_k]

            results.append({
                "position": mask_pos_idx,
                "candidates": candidates
            })

        return results

    def evaluate_mlm_score_fast(self, text, skip_positions=None):
        """文全体のMLMスコアを計算"""
        if skip_positions is None:
            skip_positions = set()
        
        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=MAX_TOKENS)
        
        if not self._use_token_type_ids:
            inputs.pop('token_type_ids', None)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
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
        MAX_BATCH = MLM_SCORE_MAX_BATCH

        all_logits = []
        for batch_start in range(0, batch_size, MAX_BATCH):
            batch_end = min(batch_start + MAX_BATCH, batch_size)
            batch_positions = eval_positions[batch_start:batch_end]
            
            mini_batch = input_ids.repeat(len(batch_positions), 1)
            for idx, pos in enumerate(batch_positions):
                mini_batch[idx, pos] = self.tokenizer.mask_token_id
            
            mini_kwargs = {}
            if "attention_mask" in inputs:
                mini_kwargs["attention_mask"] = inputs["attention_mask"].repeat(len(batch_positions), 1)
            if "token_type_ids" in inputs:
                mini_kwargs["token_type_ids"] = inputs["token_type_ids"].repeat(len(batch_positions), 1)
            
            with torch.no_grad():
                outputs = self.model(mini_batch, **mini_kwargs)
                if isinstance(outputs, tuple):
                    all_logits.append(outputs[0].cpu())
                else:
                    all_logits.append(outputs.logits.cpu())
        
        logits = torch.cat(all_logits, dim=0)

        total_score = 0.0
        for idx, pos in enumerate(eval_positions):
            original_token = input_ids[0, pos].item()
            token_logits = logits[idx, pos]
            prob = torch.softmax(token_logits, dim=0)[original_token].item()
            total_score += math.log(prob + PLL_LOG_PROB_EPS)
        
        return total_score


# ==================== 512トークン超対応 ====================
def predict_masks_per_window(model, text, jamo_info=None, use_full_context=True):
    """
    各MASK位置ごとに個別のコンテキストウィンドウで予測
    512トークン以上離れたMASK対応
    """
    text = text.replace("[MASK]", model.mask_token)  # モデル固有のmask_tokenに変換
    full_tokens = model.tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"][0].tolist()
    mask_token_id = model.tokenizer.mask_token_id
    
    mask_positions_in_tokens = [
        i for i, tid in enumerate(full_tokens) if tid == mask_token_id
    ]
    
    if not mask_positions_in_tokens:
        return []
    
    # 全MASKが512トークン内に収まるか確認
    total_span = mask_positions_in_tokens[-1] - mask_positions_in_tokens[0] + 1
    if total_span + 2 <= MAX_TOKENS:
        return model.predict_masks(text, jamo_info=jamo_info, use_full_context=use_full_context)
    
    # MASKが離れすぎ → 各MASK位置ごとに個別ウィンドウ
    jamo_map = {}
    if jamo_info:
        for ji in jamo_info:
            jamo_map[ji[0]] = ji[1]
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
            if model._use_token_type_ids:
                token_type_ids = torch.zeros_like(input_ids)
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model.model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits
        
        mask_logits = logits[0, local_mask_pos + 1]  # +1 for CLS token
        scaled_logits = mask_logits / TEMPERATURE

        jamo_pattern = jamo_map.get(mask_idx)

        if jamo_pattern:
            target_ids = model.hangul_ids_tensor
            top_k = min(200, len(target_ids))
        else:
            target_ids = model.valid_ids_tensor
            top_k = min(100, len(target_ids))
        
        target_logits = scaled_logits[target_ids]
        top_vals, top_indices = torch.topk(target_logits, top_k)
        top_ids = target_ids[top_indices]
        top_probs = torch.softmax(top_vals, dim=0)
        
        candidates = []
        for i, p in zip(top_ids, top_probs):
            token_id = i.item()
            if token_id in model.tokenizer.all_special_ids:
                continue

            token = model.id2token_cache.get(token_id, model.unk_token)
            token_clean = token.lstrip('▁ĠĊ▃')
            if token_clean.startswith('##'):
                token_clean = token_clean[2:]
            if not token_clean:
                continue

            if jamo_pattern:
                if not re.fullmatch(r'[\uAC00-\uD7A3]+', token_clean):
                    continue
                if any(ord(c) in range(0x1100, 0x1200) or ord(c) in range(0x3130, 0x3190) for c in token_clean):
                    continue
                if not match_jamo_pattern(token_clean, jamo_pattern):
                    continue
            else:
                if not model.is_valid_korean_token(token):
                    continue

            candidates.append({
                'token': token_clean,
                'probability': p.item()
            })
            if len(candidates) >= model.top_k:
                break

        # 確率正規化
        if candidates:
            total_prob = sum(c["probability"] for c in candidates)
            if total_prob > 0:
                for c in candidates:
                    c["probability"] /= total_prob

        results.append({
            'position': mask_idx,
            'candidates': candidates
        })

    return results


# ==================== アンサンブル計算 ====================
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


# ==================== ウィンドウPLL評価関数（中国語版から移植） ====================
def evaluate_window_pll(model, filled_text, mask_char_positions):
    """MASK位置周辺の適応ウィンドウでPLLスコアを計算（位置別呼び出し対応）"""
    use_offsets = getattr(model.tokenizer, "is_fast", False) and model.tokenizer.is_fast
    
    tokenizer_kwargs = {
        "return_tensors": "pt",
        "truncation": True,
        "max_length": MAX_TOKENS,
    }
    if use_offsets:
        tokenizer_kwargs["return_offsets_mapping"] = True
    
    inputs = model.tokenizer(filled_text, **tokenizer_kwargs)
    
    offsets = None
    if "offset_mapping" in inputs:
        offsets = inputs.pop("offset_mapping")[0].cpu().tolist()
    
    if not model._use_token_type_ids:
        inputs.pop('token_type_ids', None)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"].clone()
    attn_mask = inputs.get("attention_mask")
    token_type = inputs.get("token_type_ids")
    seq_len = input_ids.size(1)
    
    special_ids = set([
        model.tokenizer.cls_token_id,
        model.tokenizer.sep_token_id,
        model.tokenizer.pad_token_id
    ])

    # 全MASK文字位置からトークン位置を特定
    mask_token_positions = []
    if offsets is not None:
        for tidx, (start, end) in enumerate(offsets):
            if any(start <= mpos < end for mpos in mask_char_positions):
                mask_token_positions.append(tidx)

    if offsets is None or not mask_token_positions:
        # fast tokenizer が使えない場合や位置特定に失敗した場合は中央を使用
        # （比率推定はBPE/SentencePieceで誤差が大きいため不採用）
        center_pos = seq_len // 2
        mask_token_positions = [center_pos]

    mask_token_positions_set = set(mask_token_positions)

    center_pos = (sum(mask_token_positions) // len(mask_token_positions)
                  if mask_token_positions else seq_len // 2)

    # 適応ウィンドウサイズ（シーケンス長に応じてMIN〜MAXの範囲で決定）
    adaptive_window = min(
        PLL_CONTEXT_WINDOW_MAX,
        max(PLL_CONTEXT_WINDOW_MIN, seq_len // 2)
    )

    # 短い文はそのまま全評価（候補トークン位置を除外）
    if seq_len <= adaptive_window * 2 + 2:
        score = model.evaluate_mlm_score_fast(filled_text, skip_positions=mask_token_positions_set)
        eval_count = sum(
            1 for i in range(seq_len)
            if input_ids[0, i].item() not in special_ids
            and i not in mask_token_positions_set
        )
        return score, max(eval_count, 1)

    start = max(1, center_pos - adaptive_window)
    end = min(seq_len - 1, center_pos + adaptive_window + 1)
    
    eval_positions = []
    for i in range(start, end):
        token_id = input_ids[0, i].item()
        if token_id in special_ids:
            continue
        if i in mask_token_positions_set:
            continue  # 全候補トークン位置を除外
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
        
        mini_kwargs = {}
        if attn_mask is not None:
            mini_kwargs["attention_mask"] = attn_mask.repeat(len(batch_positions), 1)
        if token_type is not None:
            mini_kwargs["token_type_ids"] = token_type.repeat(len(batch_positions), 1)
        
        with torch.no_grad():
            outputs = model.model(mini_batch, **mini_kwargs)
            if isinstance(outputs, tuple):
                all_logits.append(outputs[0].cpu())
            else:
                all_logits.append(outputs.logits.cpu())
    
    logits = torch.cat(all_logits, dim=0)
    
    total_score = 0.0
    for idx, pos in enumerate(eval_positions):
        original_token = input_ids[0, pos].item()
        token_logits = logits[idx, pos]
        prob = torch.softmax(token_logits, dim=0)[original_token].item()
        total_score += math.log(prob + PLL_LOG_PROB_EPS)
    
    return total_score, len(eval_positions)


# ==================== main ====================
def main():
    if len(sys.argv) < 2:
        print("使い方: python script.py <入力テキストファイル>")
        print("例: python script.py input.txt")
        return
    
    input_file = sys.argv[1]
    input_path = Path(input_file)
    
    # 出力ファイル名を生成
    output_path = input_path.with_name(input_path.stem + "_out" + input_path.suffix)
    output_file = str(output_path)
    
    print(f"入力ファイル: {Path(input_file).absolute()}")
    print(f"出力ファイル: {output_path.absolute()}")
    
    # 既存の出力ファイルを削除（初期化）
    if output_path.exists():
        output_path.unlink()
    
    # アンサンブル結果ファイルも削除
    for method in ENSEMBLE_METHODS:
        method_path = output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix)
        if method_path.exists():
            method_path.unlink()

    # 横並びアンサンブルファイルも削除
    ensemble_combined_path = output_path.with_name(input_path.stem + "_out_ensemble" + output_path.suffix)
    if ensemble_combined_path.exists():
        ensemble_combined_path.unlink()
    
    # モデルをロード
    models = {}
    for model_name in COMPARE_MODELS:
        model_path = f"models/{model_name.replace('/', '_')}"
        if not Path(model_path).exists():
            print(f"警告: モデルが見つかりません: {model_path} (スキップします)")
            continue
        models[model_name] = KoreanMaskCompletion(model_path, top_k=TOP_K)
    
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
        if not line or line.startswith('#'):
            continue
        
        original_line = line
        
        # 字母モード処理
        # 注意: replace_jamos_with_masksは[MASK_X]をスキップするので、
        # [MASK_C]ㄹ → [MASK_C][MASK] (jamo_infoに(0, 'ㄹ')が記録)
        jamo_info = []
        if JAMO_MODE and any(c in line for c in CHOSUNG + JUNGSUNG):
            line, jamo_info = replace_jamos_with_masks(line)
            if jamo_info:
                print(f"字母検出 → 完成形変換後処理: '{original_line}' → '{line}'")
        
        # 字母変換後のline中の全マスク位置をマッピング
        # jamo_infoは位置情報付き: [(idx, pattern, position_in_line), ...]
        # 位置ベースで字母[MASK]を特定する
        
        # 字母由来[MASK]の位置セット
        jamo_positions = {ji[2] for ji in jamo_info}  # position_in_line
        jamo_pos_to_pattern = {ji[2]: ji[1] for ji in jamo_info}
        
        all_masks = []
        mask_pattern = r'\[MASK(?:_([A-Z0-9_]+))?\]'
        
        for m in re.finditer(mask_pattern, line):
            mask_type_suffix = m.group(1)
            pos = m.start()
            
            if mask_type_suffix:
                # カスタムマスク [MASK_X]
                all_masks.append({
                    'pos': pos, 'end': m.end(),
                    'type': mask_type_suffix,
                    'jamo': None, 'is_jamo_mask': False
                })
            else:
                # 通常 [MASK] — 位置ベースで字母かどうか判定
                is_jamo = pos in jamo_positions
                jamo_pattern = jamo_pos_to_pattern.get(pos)
                all_masks.append({
                    'pos': pos, 'end': m.end(),
                    'type': 'MASK',
                    'jamo': jamo_pattern,
                    'is_jamo_mask': is_jamo
                })
        
        # 字母由来[MASK]がカスタムマスクの直後にある場合、そのカスタムマスクに字母を紐付け
        for i in range(len(all_masks) - 1):
            current = all_masks[i]
            next_mask = all_masks[i + 1]
            
            if current['type'] != 'MASK' and next_mask['is_jamo_mask']:
                # カスタムマスクの直後に字母[MASK]がある
                if current['end'] == next_mask['pos']:
                    current['jamo'] = next_mask['jamo']
                    next_mask['belongs_to_custom'] = current['type']
                    print(f"  カスタムMASK_{current['type']}に字母情報紐付: {next_mask['jamo']}")
        
        # カスタムマスクの前置字母を検出（字母+[MASK_X] パターン）
        # prefix_jamo_for_custom: {mask_type: jamo_pattern} (行内で統一されている前提)
        prefix_jamo_for_custom = {}
        prefix_jamo_pattern = re.compile(
            r'((?:[' + CHOSUNG + r'][' + JUNGSUNG + r']?[' + JONGSUNG + r']?)|(?:[' + CHOSUNG + r']))'
            r'(\[MASK_([A-Z0-9_]+)\])'
        )
        for pjm in prefix_jamo_pattern.finditer(original_line):
            raw_jamo = pjm.group(1)
            mtype = pjm.group(3)
            # 字母→完成形に変換（1文字ずつ replace_jamos_with_masks と同ロジック）
            converted_jamo = raw_jamo
            if len(raw_jamo) == 3 and raw_jamo[0] in CHOSUNG and raw_jamo[1] in JUNGSUNG and raw_jamo[2] in JONGSUNG:
                ci = CHOSUNG.index(raw_jamo[0])
                ji = JUNGSUNG.index(raw_jamo[1])
                joi = JONGSUNG.index(raw_jamo[2]) + 1
                converted_jamo = chr(0xAC00 + ci * 588 + ji * 28 + joi)
            elif len(raw_jamo) == 2 and raw_jamo[0] in CHOSUNG and raw_jamo[1] in JUNGSUNG:
                ci = CHOSUNG.index(raw_jamo[0])
                ji = JUNGSUNG.index(raw_jamo[1])
                converted_jamo = chr(0xAC00 + ci * 588 + ji * 28)
            # else: 初声のみ → そのまま
            if mtype in prefix_jamo_for_custom:
                if prefix_jamo_for_custom[mtype][0] != converted_jamo:
                    print(f"  警告: MASK_{mtype}に異なる前置字母が混在 "
                          f"(既存: {prefix_jamo_for_custom[mtype][0]}, 新規: {converted_jamo}) → 既存を優先")
            else:
                prefix_jamo_for_custom[mtype] = (converted_jamo, pjm.group(1))
                print(f"  前置字母検出: '{pjm.group(1)}'→'{converted_jamo}' → MASK_{mtype}")

        # マスクタイプごとにグルーピング
        # カスタムマスク用processed_lineを生成
        custom_mask_pattern = r'\[MASK_([A-Z0-9_]+)\]'
        custom_types_in_line = list(dict.fromkeys(re.findall(custom_mask_pattern, line)))
        
        for mask_type_key in custom_types_in_line:
            # カスタムマスクに紐付く字母[MASK]の位置を収集
            custom_jamo_mask_positions = set()
            # all_masksから、このタイプのカスタムマスクに紐付く字母[MASK]位置を特定
            # また、各カスタムマスクの位置→字母パターンのマッピングを作成
            custom_mask_pos_to_jamo = {}  # {カスタムマスクのpos: jamo_pattern}
            
            for am in all_masks:
                if am['type'] == mask_type_key and am['jamo'] is not None:
                    custom_mask_pos_to_jamo[am['pos']] = am['jamo']
                    for am2 in all_masks:
                        if am2.get('belongs_to_custom') == mask_type_key and am2['is_jamo_mask']:
                            custom_jamo_mask_positions.add(am2['pos'])
            
            # processed_lineを構築しながらjamo_infoも同時に構築
            custom_jamo_info = []
            processed_parts = []
            i = 0
            proc_mask_idx = 0
            while i < len(line):
                # カスタムマスクチェック
                cm_match = re.match(r'\[MASK_([A-Z0-9_]+)\]', line[i:])
                if cm_match:
                    cm_type = cm_match.group(1)
                    if cm_type == mask_type_key:
                        # 対象カスタムマスク → [MASK]に置換
                        processed_parts.append('[MASK]')
                        # このマスクに字母が紐付いているか確認
                        jamo_for_this = custom_mask_pos_to_jamo.get(i)
                        if jamo_for_this is not None:
                            custom_jamo_info.append((proc_mask_idx, jamo_for_this))
                        proc_mask_idx += 1
                    # 他のカスタムマスク → 除去
                    i += len(cm_match.group())
                    continue
                
                # 通常[MASK]チェック
                if line[i:i+6] == '[MASK]':
                    pos = i
                    if pos in custom_jamo_mask_positions:
                        # このカスタムに紐付く字母[MASK] → 除去（統合済み）
                        pass
                    else:
                        # 通常MASKまたは他に紐付くMASK → 除去
                        pass
                    i += 6
                    continue
                
                processed_parts.append(line[i])
                i += 1
            
            processed_line = ''.join(processed_parts)
            
            # 前置字母がある場合: processed_line から前置字母文字列を除去し、
            # jamo_info に前置字母を設定する
            prefix_jamo_entry = prefix_jamo_for_custom.get(mask_type_key)
            if prefix_jamo_entry is not None:
                prefix_jamo, raw_jamo = prefix_jamo_entry
                def remove_prefix_jamo_from_processed(pl, raw_jamo_str):
                    """[MASK]の直前にある字母文字を除去"""
                    import re as _re
                    escaped = _re.escape(raw_jamo_str)
                    return _re.sub(escaped + r'(\[MASK\])', r'\1', pl)
                processed_line = remove_prefix_jamo_from_processed(processed_line, raw_jamo)

                num_masks_in_line = processed_line.count('[MASK]')
                custom_jamo_info = [(midx, prefix_jamo) for midx in range(num_masks_in_line)]

                group_key = f"{mask_type_key}:{prefix_jamo}"
                print(f"  前置字母グループキー: '{group_key}', processed: '{processed_line}', jamo_info: {custom_jamo_info}")
            else:
                group_key = mask_type_key

            if group_key not in mask_groups:
                mask_groups[group_key] = []
            mask_groups[group_key].append({
                'line_num': line_num,
                'original': original_line,
                'processed': processed_line,
                'jamo_info': custom_jamo_info,
                'mask_type_display': mask_type_key  # 表示用に元のタイプを保持
            })
            if custom_jamo_info:
                print(f"  MASK_{mask_type_key} processed: '{processed_line}', jamo_info: {custom_jamo_info}")
        
        # 通常[MASK]の処理
        # 通常[MASK]のうち、カスタムマスクに紐付く字母[MASK]を除外
        has_normal_mask = any(
            am['type'] == 'MASK' and not am.get('belongs_to_custom')
            for am in all_masks
        )
        
        if has_normal_mask:
            # カスタムマスクと、カスタムに紐付く字母[MASK]を除去したprocessed_lineを構築
            processed_parts = []
            i = 0
            normal_jamo_info = []
            proc_normal_mask_idx = 0
            
            # line中をスキャンしてprocessed_lineを構築
            while i < len(line):
                # カスタムマスクチェック
                cm_match = re.match(r'\[MASK_[A-Z0-9_]+\]', line[i:])
                if cm_match:
                    # カスタムマスクは除去
                    i += len(cm_match.group())
                    continue
                
                # 通常[MASK]チェック
                if line[i:i+6] == '[MASK]':
                    # この[MASK]がカスタムに紐付く字母[MASK]かチェック
                    is_custom_jamo = False
                    for am in all_masks:
                        if (am['type'] == 'MASK' and 
                            am['pos'] == i and 
                            am.get('belongs_to_custom')):
                            is_custom_jamo = True
                            break
                    
                    if is_custom_jamo:
                        # カスタムに紐付く字母[MASK] → 除去
                        i += 6
                        continue
                    else:
                        # 通常MASK → 残す
                        processed_parts.append('[MASK]')
                        # 字母情報の紐付け
                        for am in all_masks:
                            if (am['type'] == 'MASK' and 
                                am['pos'] == i and 
                                am['jamo'] is not None):
                                normal_jamo_info.append((proc_normal_mask_idx, am['jamo']))
                                break
                        proc_normal_mask_idx += 1
                        i += 6
                        continue
                
                processed_parts.append(line[i])
                i += 1
            
            processed_line = ''.join(processed_parts)
            
            if 'MASK' not in mask_groups:
                mask_groups['MASK'] = []
            mask_groups['MASK'].append({
                'line_num': line_num,
                'original': original_line,
                'processed': processed_line,
                'jamo_info': normal_jamo_info
            })
            if normal_jamo_info:
                print(f"  通常MASK processed: '{processed_line}', jamo_info: {normal_jamo_info}")

    # ===== ステップ2: グループごとに処理 =====
    all_results = []
    process_count = 0
    

    for mask_type, entries in mask_groups.items():
        # ============================================================
        # 通常MASKの処理ブロック
        # ============================================================
        if mask_type == 'MASK':
            for entry in entries:
                process_count += 1
                
                group_lines = [(entry['line_num'], entry['original'])]
                combined_original = entry['original']
                combined_processed = entry['processed']
                entry_jamo_info = entry.get('jamo_info', [])
                
                # 字母マップ構築（mask_idx -> pattern）
                entry_jamo_map = {}
                if entry_jamo_info:
                    for ji in entry_jamo_info:
                        entry_jamo_map[ji[0]] = ji[1]
                
                if entry_jamo_info:
                    print(f"\n処理中: 通常字母（単一行）: {process_count}番目")
                else:
                    print(f"\n処理中: 通常MASK（単一行）: {process_count}番目")
                print(f"対象行: 行{entry['line_num']}")
                if entry_jamo_info:
                    print(f"字母情報: {entry_jamo_info}")
                
                # 各モデルで予測
                all_model_predictions = {}
                for model_name, model in models.items():
                    predictions = predict_masks_per_window(
                        model, combined_processed, 
                        jamo_info=entry_jamo_info,
                        use_full_context=USE_FULL_LINE_CONTEXT
                    )
                    all_model_predictions[model_name] = predictions
                
                # 全MASKについて表示
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
                        mask_type='MASK',
                        group_lines=group_lines,
                        mask_index=mask_idx,
                        show_header=(mask_idx == 0),
                        f=None,
                        jamo_map=entry_jamo_map
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
                            mask_type='MASK',
                            group_lines=group_lines,
                            mask_index=mask_idx,
                            show_header=(mask_idx == 0),
                            f=f,
                            jamo_map=entry_jamo_map
                        )
                
                # アンサンブル計算（3方式）
                for method in ENSEMBLE_METHODS:
                    if method == "rank":
                        ensemble_results = calculate_ensemble_scores_rank(all_model_predictions, list(models.keys()))
                    elif method == "probability":
                        ensemble_results = calculate_ensemble_scores_probability(all_model_predictions, list(models.keys()))
                    elif method == "hybrid":
                        ensemble_results = calculate_ensemble_scores_hybrid(all_model_predictions, list(models.keys()))
                    
                    display_ensemble_results(
                        ensemble_results=ensemble_results,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        original_text=entry['original'],
                        mask_type='MASK',
                        group_lines=group_lines,
                        method=method,
                        jamo_map=entry_jamo_map
                    )
                    
                    output_file_method = str(output_path.with_name(output_path.stem + f"_{method}" + output_path.suffix))
                    save_ensemble_results(
                        output_file=output_file_method,
                        ensemble_results=ensemble_results,
                        line_num=entry['line_num'],
                        process_count=process_count,
                        original_text=entry['original'],
                        mask_type='MASK',
                        group_lines=group_lines,
                        method=method,
                        jamo_map=entry_jamo_map
                    )
                
                all_results.append((entry['line_num'], process_count, combined_original, all_model_predictions, list(models.keys())))

                # 3方式横並び出力
                results_by_method = {}
                for method in ENSEMBLE_METHODS:
                    if method == "rank":
                        results_by_method[method] = calculate_ensemble_scores_rank(all_model_predictions, list(models.keys()))
                    elif method == "probability":
                        results_by_method[method] = calculate_ensemble_scores_probability(all_model_predictions, list(models.keys()))
                    elif method == "hybrid":
                        results_by_method[method] = calculate_ensemble_scores_hybrid(all_model_predictions, list(models.keys()))

                ensemble_combined_file = str(output_path.with_name(input_path.stem + "_out_ensemble" + output_path.suffix))
                display_and_save_ensemble_combined(
                    output_file=ensemble_combined_file,
                    results_by_method=results_by_method,
                    line_num=entry['line_num'],
                    process_count=process_count,
                    original_text=entry['original'],
                    mask_type='MASK',
                    group_lines=group_lines,
                    jamo_map=entry_jamo_map
                )
        
        # ============================================================
        # カスタムマスクの処理ブロック（PLL方式）
        # ============================================================
        else:
            process_count += 1

            line_nums = [entry['line_num'] for entry in entries]
            group_lines = [(entry['line_num'], entry['original']) for entry in entries]

            # "TYPE:jamo" 形式のキーから表示用 mask_type を取得
            display_mask_type = entries[0].get('mask_type_display', mask_type.split(':')[0])

            if len(entries) >= 2:
                print(f"\n処理中: MASK_{display_mask_type}グループ（{len(entries)}行連結）: {process_count}番目")
            else:
                print(f"\n処理中: MASK_{display_mask_type}（単一行）: {process_count}番目")
            print(f"対象行: " + ", ".join(f"行{num}" for num in line_nums))

            combined_original = '\n'.join(entry['original'] for entry in entries)

            mask_counts = [entry['processed'].count('[MASK]') for entry in entries]
            total_masks = sum(mask_counts)

            per_entry_rescored = {}

            if len(entries) >= 2 or total_masks >= 2:
                # ============================================================
                # Top-k組み合わせPLL方式によるマスクグループ統合
                # ============================================================
                
                # PLL評価用モデルの選択
                eval_model_name = 'klue/roberta-large' # リスト中で最大のモデル（large）なので、PLL評価の精度が最も高いと期待されるため
                if eval_model_name not in models:
                    eval_model_name = next(iter(models))
                eval_model = models[eval_model_name]
                
                # ===== ステップ1: 全モデルから候補プールを収集（MASK位置ごと） =====
                all_mask_candidates_by_position = {}
                
                for entry in entries:
                    for model_name, model in models.items():
                        preds = predict_masks_per_window(
                            model,
                            entry['processed'],
                            jamo_info=entry.get('jamo_info', []),
                            use_full_context=USE_FULL_LINE_CONTEXT
                        )
                        if not preds:
                            continue
                        for pred in preds:
                            mask_idx = pred['position']
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
                # 位置別の字母パターンを収集
                jamo_per_mask_idx = {}
                for entry in entries:
                    for ji in entry.get('jamo_info', []):
                        midx = ji[0]
                        pattern = ji[1]
                        if midx not in jamo_per_mask_idx:
                            jamo_per_mask_idx[midx] = []
                        jamo_per_mask_idx[midx].append(pattern)
                
                candidate_pool_filtered_by_position = {}
                for mask_idx, candidates_dict in all_mask_candidates_by_position.items():
                    sorted_candidates = sorted(
                        candidates_dict.items(),
                        key=lambda x: x[1]['prob_sum'] / x[1]['count'],
                        reverse=True
                    )
                    jamo_patterns = jamo_per_mask_idx.get(mask_idx, [])
                    filtered = {}
                    for token, data in sorted_candidates[:CANDIDATE_POOL_TOP_K]:
                        if not KoreanMaskCompletion.is_valid_korean_token(token):
                            continue
                        if jamo_patterns and not any(match_jamo_pattern(token, jp) for jp in jamo_patterns):
                            continue
                        filtered[token] = data
                    if not filtered:
                        # フォールバック: 字母フィルタなし
                        for token, data in sorted_candidates[:CANDIDATE_POOL_TOP_K]:
                            if KoreanMaskCompletion.is_valid_korean_token(token):
                                filtered[token] = data
                    candidate_pool_filtered_by_position[mask_idx] = filtered
                
                # ===== ステップ3: 組み合わせ候補を生成してPLL評価 =====
                num_masks = (max(all_mask_candidates_by_position.keys()) + 1
                             if all_mask_candidates_by_position else 0)
                
                if num_masks == 0:
                    print("警告: 候補プールが空です。フォールバック処理")
                    all_model_predictions = {}
                    for model_name, model in models.items():
                        all_model_predictions[model_name] = predict_masks_per_window(
                            model, entries[0]['processed'],
                            jamo_info=entries[0].get('jamo_info', []),
                            use_full_context=USE_FULL_LINE_CONTEXT
                        )
                    unified_candidates = []
                else:
                    mask_candidate_lists = []
                    for mask_idx in range(num_masks):
                        candidates = candidate_pool_filtered_by_position.get(mask_idx, {})
                        top_candidates = sorted(
                            candidates.items(),
                            key=lambda x: x[1]['prob_sum'] / x[1]['count'],
                            reverse=True
                        )[:10]
                        mask_candidate_lists.append([token for token, _ in top_candidates])
                    
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
                    pll_cache = {}
                    
                    for combo in combinations:
                        combo_key = tuple(combo)
                        
                        log_prob_sum = 0.0
                        for mask_idx, token in enumerate(combo):
                            token_data = candidate_pool_filtered_by_position[mask_idx].get(token)
                            if token_data:
                                log_prob_sum += math.log(
                                    token_data['prob_sum'] / token_data['count'] + PLL_LOG_PROB_EPS
                                )
                        avg_log_prob = log_prob_sum / len(combo)
                        
                        combination_scores[combo_key] = {
                            'pll': 0.0,
                            'eval_count': 0,
                            'avg_log_prob': avg_log_prob,
                            'pll_per_entry': [0.0] * len(entries),
                            'eval_per_entry': [0] * len(entries)
                        }
                        
                        for entry_idx, entry in enumerate(entries):
                            clean_processed = entry['processed']
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
                                        eval_model, filled, [adjusted_pos]
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
                    
                    # per_entry_rescoredを行ごとに計算（複数エントリの場合）
#                   if num_masks == 1 and len(entries) >= 2:
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
                    
                    # ===== ステップ4A: 統一PLL（combination_scoresから集計） =====
                    unified_rescored = []
                    for combo, scores in combination_scores.items():
                        token = combo[0] if num_masks == 1 else ''.join(combo)
                        normalized_pll = scores['pll'] / max(scores['eval_count'], 1)
                        unified_rescored.append({
                            'token': token,
                            'normalized_pll': normalized_pll,
                            'avg_log_prob': scores['avg_log_prob'],
                            'pll_score': scores['pll'],
                            'eval_count': scores['eval_count']
                        })
                    
                    if unified_rescored:
                        for item in unified_rescored:
                            item['hybrid_log'] = (PLL_HYBRID_ALPHA * item['normalized_pll']
                                                  + PLL_HYBRID_BETA  * item['avg_log_prob'])
                        unified_rescored.sort(key=lambda x: -x['hybrid_log'])
                        max_log = unified_rescored[0]['hybrid_log']
                        exp_scores = [math.exp((x['hybrid_log'] - max_log) / PLL_TEMPERATURE) for x in unified_rescored]
                        total = sum(exp_scores)
                        for i, item in enumerate(unified_rescored):
                            item['probability'] = exp_scores[i] / total if total > 0 else 0.0
                    
                    unified_candidates = unified_rescored[:TOP_K]
                    
                    # ===== ステップ5: 全モデルに統一候補を適用 =====
                    all_model_predictions = {}
                    if not unified_candidates:
                        print("警告: 統一候補の生成に失敗。フォールバック処理")
                        for model_name, model in models.items():
                            predictions = model.predict_masks(
                                entries[0]['processed'],
                                jamo_info=entries[0].get('jamo_info', []),
                                use_full_context=USE_FULL_LINE_CONTEXT
                            )
                            all_model_predictions[model_name] = predictions if predictions else []
                    else:
                        for model_name in models.keys():
                            all_model_predictions[model_name] = [
                                {'position': 0, 'candidates': unified_candidates}
                            ]
                
            else:
                # 通常処理（単一MASK）
                combined_processed = '\n'.join(entry['processed'] for entry in entries)
                all_model_predictions = {}
                for model_name, model in models.items():
                    all_model_predictions[model_name] = predict_masks_per_window(
                        model, combined_processed,
                        jamo_info=entries[0].get('jamo_info', []),
                        use_full_context=USE_FULL_LINE_CONTEXT
                    )

            # 全MASKについて表示
            pred_lengths = [len(preds) for preds in all_model_predictions.values() if len(preds) > 0]
            num_masks = max(pred_lengths) if pred_lengths else 0
            model_abbr = [MODEL_ABBREVIATIONS.get(name, name) for name in model_names]
            is_unified = (len(entries) >= 2 or total_masks >= 2)
            
            for mask_idx in range(num_masks):
                if is_unified and len(entries) >= 2 and per_entry_rescored:
                    # PLL方式: 統一PLL + 行別PLL を表示
                    display_labels = ["統一PLL"] + [f"行{entry['line_num']}" for entry in entries]
                    predictions_for_display = []
                    
                    # 統一PLL結果
                    preds = all_model_predictions[model_names[0]]
                    if mask_idx < len(preds):
                        candidates = preds[mask_idx].get('candidates', [])
                        predictions_for_display.append([
                            (c['token'], c['probability'])
                            for c in candidates[:TOP_K]
                        ])
                    else:
                        predictions_for_display.append([])
                    
                    # 行別PLL結果
                    for entry_idx in range(len(entries)):
                        candidates = per_entry_rescored.get(entry_idx, [])
                        predictions_for_display.append([
                            (c['token'], c['probability'])
                            for c in candidates[:TOP_K]
                        ])
                
                elif is_unified:
                    # PLL方式（単一エントリ）
                    display_labels = ["PLL"]
                    predictions_for_display = []
                    preds = all_model_predictions[model_names[0]]
                    if mask_idx < len(preds):
                        candidates = preds[mask_idx].get('candidates', [])
                        predictions_for_display.append([
                            (c['token'], c['probability'])
                            for c in candidates[:TOP_K]
                        ])
                
                else:
                    # 通常表示（モデル別）
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
                    mask_type=display_mask_type,
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
                        mask_type=display_mask_type,
                        group_lines=group_lines,
                        mask_index=mask_idx,
                        show_header=(mask_idx == 0),
                        f=f
                    )

            # アンサンブル計算・表示・保存（3方式）
            for method in ENSEMBLE_METHODS:
                if method == "rank":
                    ensemble_results = calculate_ensemble_scores_rank(all_model_predictions, list(models.keys()))
                elif method == "probability":
                    ensemble_results = calculate_ensemble_scores_probability(all_model_predictions, list(models.keys()))
                elif method == "hybrid":
                    ensemble_results = calculate_ensemble_scores_hybrid(all_model_predictions, list(models.keys()))

                display_ensemble_results(
                    ensemble_results=ensemble_results,
                    line_num=line_nums[0],
                    process_count=process_count,
                    original_text=combined_original,
                    mask_type=display_mask_type,
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
                    mask_type=display_mask_type,
                    group_lines=group_lines,
                    method=method
                )
            
            all_results.append((line_nums[0], process_count, combined_original, all_model_predictions, list(models.keys())))

            # 3方式横並び出力
            results_by_method = {}
            for method in ENSEMBLE_METHODS:
                if method == "rank":
                    results_by_method[method] = calculate_ensemble_scores_rank(all_model_predictions, list(models.keys()))
                elif method == "probability":
                    results_by_method[method] = calculate_ensemble_scores_probability(all_model_predictions, list(models.keys()))
                elif method == "hybrid":
                    results_by_method[method] = calculate_ensemble_scores_hybrid(all_model_predictions, list(models.keys()))

            ensemble_combined_file = str(output_path.with_name(input_path.stem + "_out_ensemble" + output_path.suffix))
            display_and_save_ensemble_combined(
                output_file=ensemble_combined_file,
                results_by_method=results_by_method,
                line_num=line_nums[0],
                process_count=process_count,
                original_text=combined_original,
                mask_type=display_mask_type,
                group_lines=group_lines
            )


if __name__ == "__main__":
    main()
