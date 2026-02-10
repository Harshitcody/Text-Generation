
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
    value="gpt2",
    help="Use 'gpt2' or path to your fine-tuned model"
)

max_new_tokens = st.sidebar.slider("Max New Tokens", 20, 500, 150)
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.8)
top_k = st.sidebar.slider("Top-K", 0, 100, 50)
top_p = st.sidebar.slider("Top-P", 0.1, 1.0, 0.95)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"**Device:** `{device.upper()}`")

# -----------------------------
# Title
# -----------------------------
st.title("🤖 Professional AI Text Generator")
st.markdown("Generate creative, high-quality text using a GPT-based language model.")

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(path)
    model.to(device)
    model.eval()

    return tokenizer, model

try:
    tokenizer, model = load_model(model_path, device)
except Exception as e:
    st.error(f"❌ Failed to load model:\n{e}")
    st.stop()

# -----------------------------
# Input Section
# -----------------------------
with st.form("generation_form"):
    col1, col2 = st.columns([2, 1])

    with col1:
        prompt = st.text_area(
            "Enter your prompt",
            height=200,
            placeholder="Example: Alice was walking through the forest when..."
        )

    with col2:
        st.info(
            "💡 Tips:\n"
            "- Higher temperature → more creativity\n"
            "- Lower temperature → more accuracy\n"
            "- Fine-tuned models give best results"
        )

    generate_btn = st.form_submit_button("✨ Generate Text")

# -----------------------------
# Text Generation
# -----------------------------
if generate_btn:
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt.")
    else:
        with st.spinner("Generating text..."):
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k > 0 else None,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            generated_text = tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )

        st.subheader("📄 Generated Output")
        st.write(generated_text)

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
st.markdown("Built with ❤️ using **Streamlit + Hugging Face Transformers**")


