import pandas as pd
import ast
from collections import Counter

# 1. Load tập train vừa tạo
df = pd.read_csv("multisentiment-uit-vsfc/df_final_train.csv")

# 2. Hàm chuyển đổi string thành list và đếm
all_topics = []
all_sentiments = []

for _, row in df.iterrows():
    # Chuyển string thành list
    topics = ast.literal_eval(str(row["topic"]))
    sentiments = ast.literal_eval(str(row["sentiment"]))

    all_topics.extend(topics)
    all_sentiments.extend(sentiments)

# 3. Thống kê số lượng
topic_counts = Counter(all_topics)
sentiment_counts = Counter(all_sentiments)

# 4. In kết quả ra màn hình
print("-" * 30)
print(f"TỔNG SỐ DÒNG: {len(df)}")
print("-" * 30)

print("THỐNG KÊ TOPICS (KHÍA CẠNH):")
# Sắp xếp theo ID của topic cho dễ nhìn
for t_id in sorted(topic_counts.keys()):
    print(f"  - Topic {t_id}: {topic_counts[t_id]:>6} mẫu")

print("\nTHỐNG KÊ SENTIMENT (CẢM XÚC):")
# Ví dụ: 0: Tiêu cực, 1: Trung tính, 2: Tích cực
sentiment_map = {
    0: "Tiêu cực",
    1: "Trung tính",
    2: "Tích cực",
}  # Thay đổi theo map của bạn
for s_id in sorted(sentiment_counts.keys()):
    label = sentiment_map.get(s_id, f"Nhãn {s_id}")
    print(f"  - {label}: {sentiment_counts[s_id]:>6} mẫu")
print("-" * 30)
