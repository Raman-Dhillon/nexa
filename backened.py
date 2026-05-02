import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from bert_score import score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ---------------- SETUP ----------------
smooth = SmoothingFunction().method1
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD MODELS ----------------

# M2M (LoRA)
m2m_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
m2m_base = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
m2m_model = PeftModel.from_pretrained(m2m_base, "m2m_lora").to(device)

# NLLB (LoRA)
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_base = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = PeftModel.from_pretrained(nllb_base, "nllb_lora").to(device)

# Back translation (base NLLB)
nllb_back_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_back_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)

# Embeddings
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("✅ Models loaded")

# ---------------- FUNCTIONS ----------------

def translate_m2m(text):
    m2m_tokenizer.src_lang = "en"
    inputs = m2m_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens = m2m_model.generate(
            **inputs,
            forced_bos_token_id=m2m_tokenizer.get_lang_id("pa"),
            num_beams=5
        )

    return m2m_tokenizer.decode(tokens[0], skip_special_tokens=True)


def translate_nllb(text):
    nllb_tokenizer.src_lang = "eng_Latn"
    inputs = nllb_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("pan_Guru"),
            num_beams=5,
            max_length=128
        )

    return nllb_tokenizer.decode(tokens[0], skip_special_tokens=True)


def back_translate(text):
    nllb_back_tokenizer.src_lang = "pan_Guru"
    inputs = nllb_back_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens = nllb_back_model.generate(
            **inputs,
            forced_bos_token_id=nllb_back_tokenizer.convert_tokens_to_ids("eng_Latn"),
            num_beams=5
        )

    return nllb_back_tokenizer.decode(tokens[0], skip_special_tokens=True)


def evaluate(src, pa):
    back = back_translate(pa)

    bleu = sentence_bleu(
        [src.split()],
        back.split(),
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smooth
    )

    P, R, F1 = score([back], [src], lang="en", verbose=False)
    bert = F1.mean().item()

    emb1 = embed_model.encode(src, convert_to_tensor=True)
    emb2 = embed_model.encode(pa, convert_to_tensor=True)
    cosine = util.cos_sim(emb1, emb2).item()

    return bleu, bert, cosine, back


def adaptive_translation(text):
    m2m_out = translate_m2m(text)
    bleu, bert, cosine, back = evaluate(text, m2m_out)

    final = 0.6 * bert + 0.4 * cosine

    # Stronger decision
    if final < 0.82 or cosine < 0.45:
        nllb_out = translate_nllb(text)
        bleu, bert, cosine, back = evaluate(text, nllb_out)

        final = 0.6 * bert + 0.4 * cosine

        return nllb_out, "NLLB", bleu, bert, cosine, final, back

    return m2m_out, "M2M", bleu, bert, cosine, final, back