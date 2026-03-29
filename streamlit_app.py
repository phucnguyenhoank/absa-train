import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import load as attention_load
from preprocess import tokenizer, rdrsegmenter
from config import idx2topic, idx2sentiment
from config_test import TEST_MODEL_NAME


# =========================
# 1. Load model (cache để không load lại)
# =========================
@st.cache_resource
def load_absa_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model = attention_load.load_model(TEST_MODEL_NAME).to(DEVICE)
    model.eval()
    return model, DEVICE


model, DEVICE = load_absa_model()


@st.cache_resource
def load_bilstm_model():
    class ABSAPairPredictor:
        def __init__(self):
            self.model = load_model("absa_bilstm_pair_model.keras")
            with open("tokenizer.pkl", "rb") as f:
                self.tokenizer = pickle.load(f)

            with open("sentiment_mapping.pkl", "rb") as f:
                self.num_sentiments = len(pickle.load(f))
            with open("topic_mapping.pkl", "rb") as f:
                self.num_topics = len(pickle.load(f))

            self.max_seq_length = 100

        def predict_matrix(self, sentence):
            seq = self.tokenizer.texts_to_sequences([sentence.lower()])
            padded = pad_sequences(
                seq, maxlen=self.max_seq_length, padding="post"
            )

            y_pred = self.model.predict(padded, verbose=0)[0]
            return y_pred.reshape(self.num_topics, self.num_sentiments)

    return ABSAPairPredictor()


bilstm_model = load_bilstm_model()

# =========================
# 2. UI
# =========================
st.title("ABSA Demo - Aspect-Based Sentiment Analysis")
model_option = st.selectbox(
    "Chọn model", ["PhoBERT + Attention", "BiLSTM Pair Model"]
)

sentence = st.text_input("Nhập câu:")
threshold = st.slider(
    "Ngưỡng chọn Aspect thứ 2 (Top 2 threshold)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
)

if st.button("Phân tích") and sentence.strip():
    # =========================
    # CASE 1: PhoBERT + Attention
    # =========================
    if model_option == "PhoBERT + Attention":

        # =========================
        # 3. Preprocess
        # =========================
        segmented = " ".join(rdrsegmenter.word_segment(sentence))
        st.write(f"Segmented: `{segmented}`")

        inputs = tokenizer(segmented, return_tensors="pt").to(DEVICE)

        # =========================
        # 4. Inference
        # =========================
        with torch.no_grad():
            logits, attentions = model(**inputs, return_attentions=True)

        probs = torch.sigmoid(logits).squeeze(0)
        attentions = attentions.squeeze(0)

        # =========================
        # 5. Prediction text
        # =========================
        max_probs, sentiment_indices = torch.max(probs, dim=-1)
        top2_values, top2_indices = torch.topk(max_probs, k=2)

        st.subheader("Kết quả dự đoán")

        top1_score = top2_values[0].item()
        top1_idx = top2_indices[0].item()

        st.write(
            f"**{idx2topic[top1_idx]}**: "
            f"{idx2sentiment[sentiment_indices[top1_idx].item()]} "
            f"(Score: {top1_score:.2f})"
        )

        top2_score = top2_values[1].item()
        top2_idx = top2_indices[1].item()
        ratio = top2_score / top1_score
        st.write(f"Top2/Top1: {ratio:.2f}")
        if ratio >= threshold:
            st.write(
                f"**{idx2topic[top2_idx]}**: "
                f"{idx2sentiment[sentiment_indices[top2_idx].item()]} "
                f"(Score: {top2_score:.2f})"
            )
        # else:
        #     st.write(
        #         f"❌ Aspect thứ 2 ({idx2topic[top2_idx]}) bị loại "
        #         f"(ratio={ratio:.2f} < threshold={threshold})"
        #     )

        # =========================
        # 6. Bar chart
        # =========================
        st.subheader("Sentiment Distribution")

        probs_np = probs.cpu().numpy()

        fig1, ax1 = plt.subplots(figsize=(8, 5))

        x = np.arange(4)
        width = 0.25

        labels = ["Negative", "Neutral", "Positive"]
        colors = ["#e74c3c", "#f1c40f", "#2ecc71"]

        for i in range(3):
            ax1.bar(
                x + i * width,
                probs_np[:, i],
                width,
                label=labels[i],
                color=colors[i],
            )

        ax1.set_xticks(x + width)
        ax1.set_xticklabels(["Lecturer", "Program", "Facility", "Others"])
        ax1.set_ylim(0, 1)
        ax1.set_ylabel("Probability")
        ax1.legend()

        st.pyplot(fig1)

        # =========================
        # 7. Attention heatmap
        # =========================
        st.subheader("Attention Heatmap")

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        fig2, ax2 = plt.subplots(figsize=(12, 4))

        sns.heatmap(
            attentions.cpu().numpy(),
            cmap="YlOrRd",
            linewidths=0.5,
            linecolor="gray",
            ax=ax2,
        )

        ax2.set_yticks(range(4))
        ax2.set_yticklabels(
            ["Lecturer", "Program", "Facility", "Others"], rotation=0
        )

        ax2.set_xticks(range(len(tokens)))
        ax2.set_xticklabels(tokens, rotation=45, ha="right")

        st.pyplot(fig2)
    # =========================
    # CASE 2: BiLSTM Pair Model
    # =========================
    else:
        st.subheader("Kết quả dự đoán (BiLSTM)")

        matrix = bilstm_model.predict_matrix(sentence)
        probs = torch.tensor(matrix)  # để reuse code

        # =========================
        # Text prediction (giống model kia)
        # =========================
        max_probs, sentiment_indices = torch.max(probs, dim=-1)
        top2_values, top2_indices = torch.topk(max_probs, k=2)

        top1_score = top2_values[0].item()
        top1_idx = top2_indices[0].item()

        st.write(
            f"**{idx2topic[top1_idx]}**: "
            f"{idx2sentiment[sentiment_indices[top1_idx].item()]} "
            f"(Score: {top1_score:.2f})"
        )

        top2_score = top2_values[1].item()
        top2_idx = top2_indices[1].item()

        ratio = top2_score / top1_score
        st.write(f"Tỷ lệ Top2/Top1: {ratio:.2f}")

        if ratio >= threshold:
            st.write(
                f"**{idx2topic[top2_idx]}**: "
                f"{idx2sentiment[sentiment_indices[top2_idx].item()]} "
                f"(Score: {top2_score:.2f})"
            )
        else:
            st.write(
                f"❌ Aspect thứ 2 ({idx2topic[top2_idx]}) bị loại "
                f"(ratio={ratio:.2f} < threshold={threshold})"
            )

        # =========================
        # Bar chart (reuse 100%)
        # =========================
        st.subheader("Sentiment Distribution")

        probs_np = probs.numpy()

        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(4)
        width = 0.25

        labels = ["Negative", "Neutral", "Positive"]
        colors = ["#e74c3c", "#f1c40f", "#2ecc71"]

        for i in range(3):
            ax.bar(
                x + i * width,
                probs_np[:, i],
                width,
                label=labels[i],
                color=colors[i],
            )

        ax.set_xticks(x + width)
        ax.set_xticklabels(["Lecturer", "Program", "Facility", "Others"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.legend()

        st.pyplot(fig)
