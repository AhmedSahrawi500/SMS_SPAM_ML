from pathlib import Path

import streamlit as st
import torch

from train import SpamLSTMClassifier, encode_text


CHECKPOINT_PATH = Path("spam_lstm_checkpoint.pt")


@st.cache_resource
def load_model_and_artifacts() -> tuple[SpamLSTMClassifier, dict[str, int], int, float, dict[int, str]]:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "Checkpoint not found. Run `python train.py` first to create spam_lstm_checkpoint.pt"
        )

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model_params = checkpoint["model_params"]

    model = SpamLSTMClassifier(
        vocab_size=model_params["vocab_size"],
        embed_dim=model_params["embed_dim"],
        lstm_hidden_dim=model_params["lstm_hidden_dim"],
        fc_hidden_dim=model_params["fc_hidden_dim"],
        dropout=model_params["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    vocab = checkpoint["vocab"]
    max_len = checkpoint["max_len"]
    threshold = checkpoint.get("threshold", 0.5)
    label_map = checkpoint.get("label_map", {0: "Not Spam", 1: "Spam"})

    return model, vocab, max_len, threshold, label_map


def predict_sms(
    message: str,
    model: SpamLSTMClassifier,
    vocab: dict[str, int],
    max_len: int,
    threshold: float,
) -> tuple[str, float]:
    token_ids = encode_text(message, vocab, max_len)
    input_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    # input_tensor shape: [1, seq_len]

    with torch.no_grad():
        logits = model(input_tensor)
        # logits shape: [1]
        spam_probability = torch.sigmoid(logits).item()

    prediction = "Spam" if spam_probability >= threshold else "Not Spam"
    return prediction, spam_probability


def main() -> None:
    st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")
    st.title("SMS Spam Detection (PyTorch)")
    st.write("Type or paste an SMS message and classify it as Spam or Not Spam.")

    try:
        model, vocab, max_len, threshold, label_map = load_model_and_artifacts()
    except FileNotFoundError as error:
        st.error(str(error))
        st.stop()

    message = st.text_area("SMS Message", height=140, placeholder="Enter message text here...")
    predict_clicked = st.button("Predict")

    if predict_clicked:
        if not message.strip():
            st.warning("Please enter a non-empty message.")
            st.stop()

        predicted_label, spam_probability = predict_sms(message, model, vocab, max_len, threshold)
        confidence = spam_probability if predicted_label == "Spam" else (1.0 - spam_probability)

        if predicted_label == "Spam":
            st.error(f"Prediction: {label_map[1]}")
        else:
            st.success(f"Prediction: {label_map[0]}")

        st.write(f"Confidence: {confidence:.2%}")
        st.caption(
            f"Spam probability: {spam_probability:.4f} (threshold = {threshold:.2f})"
        )


if __name__ == "__main__":
    main()
