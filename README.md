# Aspect-Based Sentiment Analysis (ABSA) for Student Feedback

This project performs Aspect-Based Sentiment Analysis (ABSA) to classify student feedback into 3 sentiments (**Positive**, **Neutral**, **Negative**) across 4 core aspects: **Lecturer**, **Training Program**, **Facility**, and **Others**.

The model is trained on the **UIT-VSFC** dataset ([Hugging Face Link](https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback)).

## Example
* **Input:** *"giảng viên nhiệt tình trong giảng dạy, còn phòng lab thì đã cũ"* (The lecturer is enthusiastic, but the lab is outdated)
* **Output:** 
  * `Lecturer`: Positive
  * `Facility`: Negative

![Sentiment distribution](vis/sentiment_distri_demo.png)

---

## Quick Start

Follow these steps to set up and run the Streamlit demo locally:

### 1. Clone the Repository & Install Dependencies
This project uses `uv` for fast dependency management.
```bash
git clone https://github.com/phucnguyenhoank/absa-train.git
cd absa-train
uv sync
```

### 2. Download the Model
* Download the pre-trained model weights from [this link](https://drive.google.com/drive/folders/1Wb1pZZw4W2BvvKaAk39ycKSJSbAQHyVp?usp=drive_link).
* Place the downloaded file into the project root directory.

### 3. Run the Web App
```bash
streamlit run streamlit_app.py
```

---

## Model Architecture

The model utilizes an **Attention Mechanism** to focus specifically on words relevant to the sentiment of each individual aspect.

### Attention Visualization
The visualization demonstrates how the model assigns importance to specific tokens. For instance, the tokens `giảng_viên`, `nhiệt_tình` (enthusiastic) receive the highest attention weights when predicting the sentiment for the `Lecturer` aspect. Similarly, the tokens `lab`, `cũ` are highly weighted when determining the sentiment for the `Facility` aspect.

![Attention Mechanism](vis/attention_demo.png)


### Pipeline Overview
![Model Architecture](vis/MultiHeadSigmoid-2026-03-29-054222.png)

---

## Current Limitations

* **Aspect Limit:** Currently optimized to handle a maximum of 2 aspects per input sentence.
* **Overfitting:** The model experiences slight overfitting due to programmatic data augmentation, which was used to expand the original single-aspect, single-sentiment dataset.
