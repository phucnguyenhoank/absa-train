"""
ABSA Model Loader & Predictor
Load trained model from .h5 file and make predictions
"""

import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


class ABSAPairPredictor:
    """Load và sử dụng model đã train để predict"""

    def __init__(
        self,
        model_path,
        tokenizer_path,
        sentiment_mapping_path,
        topic_mapping_path,
        max_seq_length=100,
    ):
        """
        Load model và các components cần thiết

        Args:
            model_path: đường dẫn file .h5
            tokenizer_path: đường dẫn file tokenizer.pkl
            sentiment_mapping_path: đường dẫn file sentiment_mapping.pkl
            topic_mapping_path: đường dẫn file topic_mapping.pkl
            max_seq_length: độ dài câu khi pad (phải giống khi train)
        """
        print("=" * 70)
        print("LOADING TRAINED MODEL AND COMPONENTS")
        print("=" * 70)

        # Load model
        self.model = load_model(model_path)
        print(f"✓ Model loaded from {model_path}")

        # Load tokenizer
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        print(f"✓ Tokenizer loaded from {tokenizer_path}")

        # Load mappings
        with open(sentiment_mapping_path, "rb") as f:
            self.sentiment_mapping = pickle.load(f)
        print(f"✓ Sentiment mapping loaded from {sentiment_mapping_path}")

        with open(topic_mapping_path, "rb") as f:
            self.topic_mapping = pickle.load(f)
        print(f"✓ Topic mapping loaded from {topic_mapping_path}")

        # Store metadata
        self.max_seq_length = max_seq_length
        self.num_sentiments = len(self.sentiment_mapping)
        self.num_topics = len(self.topic_mapping)
        self.num_pairs = self.num_sentiments * self.num_topics

        print(f"\n[Model Metadata]")
        print(f"  Max sequence length: {self.max_seq_length}")
        print(f"  Number of sentiments: {self.num_sentiments}")
        print(f"  Number of topics: {self.num_topics}")
        print(f"  Total pairs: {self.num_pairs}")
        print(f"  Sentiment mapping: {self.sentiment_mapping}")
        print(f"  Topic mapping: {self.topic_mapping}")
        print("=" * 70 + "\n")

    def unflatten_pairs(self, flat_pairs, threshold=0.5):
        """
        Convert flat probability vector thành list of (topic, sentiment) pairs

        Args:
            flat_pairs: (num_pairs,) - probability vector từ model
            threshold: ngưỡng để filter

        Returns:
            List of dicts với keys: topic, sentiment, confidence
        """
        pairs = []
        for topic_idx in range(self.num_topics):
            for sent_idx in range(self.num_sentiments):
                flat_idx = topic_idx * self.num_sentiments + sent_idx
                confidence = flat_pairs[flat_idx]

                if confidence > threshold:
                    pairs.append(
                        {
                            "topic_idx": topic_idx,
                            "topic_name": self.topic_mapping[topic_idx],
                            "sentiment_idx": sent_idx,
                            "sentiment_name": self.sentiment_mapping[sent_idx],
                            "confidence": confidence,
                        }
                    )

        return pairs

    def predict_one(self, sentence, threshold=0.5, show_all=False):
        """
        Predict cho 1 câu

        Args:
            sentence: câu văn bản cần predict
            threshold: ngưỡng để lọc pairs
            show_all: có hiển thị tất cả xác suất hay không

        Returns:
            dict với keys: sentence, pairs, raw_predictions, binary_vector
        """
        print(f"\n{'='*80}")
        print(f"PREDICTION FOR: {sentence}")
        print(f"{'='*80}")

        # Tokenize và pad
        seq = self.tokenizer.texts_to_sequences([sentence.lower()])
        padded = pad_sequences(seq, maxlen=self.max_seq_length, padding="post")

        # Predict
        y_pred = self.model.predict(padded, verbose=0)[0]

        # Convert to binary
        y_pred_binary = (y_pred > threshold).astype(int)

        # Unflatten pairs
        pairs = self.unflatten_pairs(y_pred, threshold)

        # Print results
        print(
            f"\n✓ Predicted Pairs ({len(pairs)} pairs above threshold {threshold}):"
        )
        if len(pairs) > 0:
            for pair in pairs:
                print(
                    f"   ({pair['topic_name']:<20}, {pair['sentiment_name']:<12}): {pair['confidence']:.4f}"
                )
        else:
            print("   No pairs found above threshold!")

        # Show all probabilities if requested
        if show_all:
            print(f"\n[All Pair Probabilities]")
            print(
                f"{'Index':<8} {'Pair':<45} {'Probability':<12} {'Binary':<8}"
            )
            print(f"{'-'*73}")

            for pair_idx in range(self.num_pairs):
                topic_idx = pair_idx // self.num_sentiments
                sent_idx = pair_idx % self.num_sentiments

                topic_name = self.topic_mapping[topic_idx]
                sentiment_name = self.sentiment_mapping[sent_idx]
                pair_name = f"({topic_name}, {sentiment_name})"

                prob = y_pred[pair_idx]
                binary = y_pred_binary[pair_idx]

                marker = " <---" if binary == 1 else ""
                print(
                    f"{pair_idx:<8} {pair_name:<45} {prob:10.4f}  {binary:<8}{marker}"
                )

        return {
            "sentence": sentence,
            "pairs": pairs,
            "raw_predictions": y_pred,
            "binary_vector": y_pred_binary,
            "num_pairs": len(pairs),
        }

    def predict_batch(self, sentences, threshold=0.5):
        """
        Predict cho nhiều câu cùng lúc

        Args:
            sentences: list of strings
            threshold: ngưỡng để lọc

        Returns:
            List of results
        """
        results = []
        for sentence in sentences:
            result = self.predict_one(sentence, threshold, show_all=False)
            results.append(result)

        return results

    def predict_from_file(self, filepath, threshold=0.5, output_path=None):
        """
        Predict từ file CSV/TXT

        Args:
            filepath: đường dẫn file input
            threshold: ngưỡng để lọc
            output_path: nếu có, export kết quả ra CSV

        Returns:
            DataFrame với kết quả
        """
        import pandas as pd

        # Load file
        if filepath.endswith(".csv"):
            df = pd.read_csv(filepath)
        elif filepath.endswith(".txt"):
            df = pd.read_csv(filepath, sep="\t")
        else:
            raise ValueError("File must be .csv or .txt")

        print(f"\n✓ Loaded {len(df)} sentences from {filepath}")

        # Predict cho tất cả
        results = []
        for idx, row in df.iterrows():
            sentence = row["sentence"]
            result = self.predict_one(sentence, threshold, show_all=False)

            # Format pairs
            pairs_str = (
                "|".join(
                    [
                        f"({p['topic_name']}, {p['sentiment_name']}, {p['confidence']:.4f})"
                        for p in result["pairs"]
                    ]
                )
                if result["pairs"]
                else "None"
            )

            results.append(
                {
                    "sentence": sentence,
                    "predicted_pairs": pairs_str,
                    "num_pairs": result["num_pairs"],
                }
            )

        results_df = pd.DataFrame(results)

        # Export nếu có output_path
        if output_path:
            results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"✓ Results exported to {output_path}")

        return results_df

    def print_model_info(self):
        """In thông tin chi tiết về model"""
        print("\n" + "=" * 80)
        print("MODEL ARCHITECTURE")
        print("=" * 80)
        self.model.summary()


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    # 1. Load model và components
    predictor = ABSAPairPredictor(
        model_path="absa_bilstm_pair_model.keras",
        tokenizer_path="tokenizer.pkl",
        sentiment_mapping_path="sentiment_mapping.pkl",
        topic_mapping_path="topic_mapping.pkl",
        max_seq_length=100,
    )

    # 2. Predict cho 1 câu (show all probabilities)
    print("\n[EXAMPLE 1: Single Sentence Prediction]")
    result = predictor.predict_one(
        "giáo viên vui tính, tận tâm", threshold=0.5, show_all=True
    )

    # 3. Predict cho nhiều câu
    print("\n\n[EXAMPLE 2: Batch Prediction]")
    sample_sentences = [
        "giáo trình chưa cụ thể",
        "giảng buồn ngủ",
        "giáo viên vui tính, tận tâm",
        "ví dụ phù hợp với nội dung kiến thức",
    ]

    results = predictor.predict_batch(sample_sentences, threshold=0.5)

    print("\n\nBatch Results Summary:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['sentence']}")
        print(f"   Pairs found: {result['num_pairs']}")
        for pair in result["pairs"]:
            print(
                f"     - ({pair['topic_name']}, {pair['sentiment_name']}): {pair['confidence']:.4f}"
            )
        print()
    # 4. Predict từ file (uncomment if have file)
    # print("\n\n[EXAMPLE 3: Predict from File]")
    # results_df = predictor.predict_from_file(
    #     filepath='test_sentences.csv',
    #     threshold=0.5,
    #     output_path='predictions_output.csv'
    # )
    # print(results_df.head(10))

    # 5. Show model info
    # predictor.print_model_info()
