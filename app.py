import streamlit as st
from backened import adaptive_translation

st.title("🌐 NEXA Adaptive Translator")

text = st.text_area("Enter English text:")

if st.button("Translate"):
    if text:
        output, model, bleu, bert, cosine, final, back = adaptive_translation(text)

        st.subheader("📌 Translation")
        st.write(output)

        st.subheader("🤖 Model Used")
        st.write(model)

        st.subheader("📊 Scores")
        st.write(f"BLEU: {bleu:.4f}")
        st.write(f"BERTScore: {bert:.4f}")
        st.write(f"Cosine: {cosine:.4f}")
        st.write(f"Final Score: {final:.4f}")

        st.subheader("🔁 Back Translation")
        st.write(back)