import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Text Generator",
    page_icon="🤖",
    layout="wide"
)

# -----------------------------
# Sidebar Settings
# -----------------------------
st.sidebar.title("⚙️ Settings")

model_path = st.sidebar.text_input(
    "Model Path",
    value="gpt2"   # Change to "./results" if using fine-tuned model
)

max_length = st.sidebar.slider("Max Length", 50, 500, 150)
temperature = st.sidebar.slider("Temperature (Creativity)", 0.5, 1.5, 0.8)
top_k = st.sidebar.slider("Top-K", 10, 100, 50)
top_p = st.sidebar.slider("Top-P", 0.5, 1.0, 0.95)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.write(f"Device: **{device.upper()}**")

# -----------------------------
# Title
# -----------------------------
st.title("🤖 Professional AI Text Generator")
st.markdown(
    "Generate creative and grammatically correct text using a GPT-based model."
)

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(path)
    model.to(device)
    model.eval()

    return tokenizer, model

# Load model safely
try:
    tokenizer, model = load_model(model_path)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# -----------------------------
# Input Area
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_area(
        "Enter your prompt:",
        height=200,
        placeholder="Example: Alice was walking through the forest when..."
    )

with col2:
    st.info(
        "Tips:\n"
        "- Higher temperature = more creative\n"
        "- Lower temperature = more accurate\n"
        "- Use your fine-tuned model for best results"
    )

# -----------------------------
# Generate Text
# -----------------------------
if st.button("✨ Generate Text", use_container_width=True):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating..."):

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            output = model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            generated_text = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

        st.subheader("Generated Output")
        st.write(generated_text)

        # Download Button
        st.download_button(
            label="📥 Download Text",
            data=generated_text,
            file_name="generated_text.txt",
            mime="text/plain"
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit + Transformers")

