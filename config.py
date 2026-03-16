SUBSET_SIZE = 64

# HYPER PARAMETERS
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 5
LEARNING_RATE = 2e-5

best_model_name = "vnsf-44"
backbone_model_name = "vinai/phobert-base-v2"
idx2topic = {
    0: "lecturer",
    1: "training_program",
    2: "facility",
    3: "others",
}
idx2sentiment = {0: "negative", 1: "neutral", 2: "positive", 3: "none"}
sentiment2idx = {v: k for k, v in idx2sentiment.items()}
