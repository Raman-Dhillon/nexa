import streamlit as st
import torch
from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from bert_score import BERTScorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from huggingface_hub import snapshot_download
import os

smooth = SmoothingFunction().method1
device = "cpu"  # force CPU for Streamlit


@st.cache_resource
def load_m2m():
    if not os.path.exists("m2m_lora"):
        snapshot_download("Raman-Dhillon/m2m-lora", local_dir="m2m_lora")

    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    base = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    model = PeftModel.from_pretrained(base, "m2m_lora").to(device)

    return tokenizer, model


@st.cache_resource
def load_nllb():
    if not os.path.exists("nllb_lora"):
        snapshot_download("Raman-Dhillon/nllb-lora", local_dir="nllb_lora")

    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    base = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    model = PeftModel.from_pretrained(base, "nllb_lora").to(device)

    return tokenizer, model


@st.cache_resource
def load_eval_models():
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cpu")

    bert_scorer = BERTScorer(
        lang="en",
        model_type="distilbert-base-uncased"
    )

    return embed_model, bert_scorer
def translate_m2m(text, tokenizer, model):
    tokenizer.src_lang = "en"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id("pa"),
            num_beams=3
        )

    return tokenizer.decode(tokens[0], skip_special_tokens=True)


def translate_nllb(text, tokenizer, model):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("pan_Guru"),
            num_beams=3,
            max_length=128
        )

    return tokenizer.decode(tokens[0], skip_special_tokens=True)
def evaluate(src, pa, embed_model, bert_scorer):

    # 🔥 ALWAYS use NLLB for back translation
    nllb_tokenizer, nllb_model = load_nllb()

    nllb_tokenizer.src_lang = "pan_Guru"
    inputs = nllb_tokenizer(pa, return_tensors="pt").to(device)

    with torch.no_grad():
        tokens = nllb_model.generate(
            **inputs,
            forced_bos_token_id=nllb_tokenizer.convert_tokens_to_ids("eng_Latn"),
            num_beams=3
        )

    back = nllb_tokenizer.decode(tokens[0], skip_special_tokens=True)

    # BLEU
    bleu = sentence_bleu(
        [src.split()],
        back.split(),
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smooth
    )

    # BERTScore
    P, R, F1 = bert_scorer.score([back], [src])
    bert = F1.mean().item()

    # Cosine
    emb1 = embed_model.encode(src, convert_to_tensor=True)
    emb2 = embed_model.encode(pa, convert_to_tensor=True)
    cosine = util.cos_sim(emb1, emb2).item()

    return bleu, bert, cosine, back
def adaptive_translation(text):

    embed_model, bert_scorer = load_eval_models()

    # Step 1: Try M2M
    m2m_tokenizer, m2m_model = load_m2m()

    m2m_out = translate_m2m(text, m2m_tokenizer, m2m_model)
    bleu, bert, cosine, back = evaluate(
        text, m2m_out,
        embed_model, bert_scorer
    )

    final = 0.6 * bert + 0.4 * cosine

    # Step 2: Switch to NLLB if needed
    if final < 0.82 or cosine < 0.45:

        nllb_tokenizer, nllb_model = load_nllb()

        nllb_out = translate_nllb(text, nllb_tokenizer, nllb_model)
        bleu, bert, cosine, back = evaluate(
            text, nllb_out,
            embed_model, bert_scorer,
            nllb_tokenizer, nllb_model
        )

        final = 0.6 * bert + 0.4 * cosine

        return nllb_out, "NLLB", bleu, bert, cosine, final, back

    return m2m_out, "M2M", bleu, bert, cosine, final, back
