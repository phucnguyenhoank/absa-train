import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ABSAPairPredictor:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        sentiment_mapping_path,
        topic_mapping_path,
        max_seq_length=100,
    ):
        # Load model và tokenizer
        self.model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        # Load mappings để lấy số lượng dòng (topic) và cột (sentiment)
        with open(sentiment_mapping_path, "rb") as f:
            self.num_sentiments = len(pickle.load(f))
        with open(topic_mapping_path, "rb") as f:
            self.num_topics = len(pickle.load(f))

        self.max_seq_length = max_seq_length

    def predict_matrix(self, sentence):
        # Tiền xử lý
        seq = self.tokenizer.texts_to_sequences([sentence.lower()])
        padded = pad_sequences(seq, maxlen=self.max_seq_length, padding="post")

        # Dự đoán
        y_pred = self.model.predict(padded, verbose=0)[0]

        # Reshape về ma trận (Aspects x Sentiments) -> (4 x 3)
        matrix = y_pred.reshape(self.num_topics, self.num_sentiments)
        return matrix


# ============================================
# CÁCH DÙNG
# ============================================
if __name__ == "__main__":
    predictor = ABSAPairPredictor(
        model_path="absa_bilstm_pair_model.keras",
        tokenizer_path="tokenizer.pkl",
        sentiment_mapping_path="sentiment_mapping.pkl",
        topic_mapping_path="topic_mapping.pkl",
    )

    sentence = "giáo viên vui tính, tận tâm"
    matrix = predictor.predict_matrix(sentence)

    print(f"\nKết quả ma trận cho câu: '{sentence}'")
    print(matrix)
