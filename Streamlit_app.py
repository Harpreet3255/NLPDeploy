import torch
import torch.nn as nn  # Ensure you import the nn module
import torch.nn.functional as F
import streamlit as st
from transformers import BertTokenizer, BertModel

# === Load Pre-trained Model and Tokenizer ===
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# === Define Classes for Tokenizing and Model ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class NonCausalChatbot(nn.Module):
    def __init__(self, bert_model, hidden_dim, num_layers, vocab_size):
        super(NonCausalChatbot, self).__init__()
        self.bert = bert_model
        self.positional_encoding = PositionalEncoding(self.bert.config.hidden_size)
        self.fc = nn.Linear(self.bert.config.hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_masks):
        outputs = self.bert(input_ids, attention_mask=attention_masks)
        hidden_states = outputs.last_hidden_state
        hidden_states = self.positional_encoding(hidden_states)
        logits = self.fc(hidden_states)
        return logits

# === Load Model from File ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 512
num_layers = 4
vocab_size = tokenizer.vocab_size

model = NonCausalChatbot(bert_model, hidden_dim, num_layers, vocab_size).to(device)
model.load_state_dict(torch.load("chatbot_model_final.pth"))
model.eval()

# === Helper Function to Generate Response ===
def generate_response(input_text, tokenizer, model, device, max_len=50, temperature=1.0, logit_clip_value=10.0):
    # Tokenize the input text
    encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoded_input.input_ids.to(device)
    attention_mask = encoded_input.attention_mask.to(device)

    model.eval()  # Set the model to evaluation mode

    # Initialize the sequence with the input tokens
    generated_sequence = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_len):
            # Forward pass through the model
            logits = model(generated_sequence, attention_mask)

            # Only take the logits of the last token
            logits = logits[:, -1, :]

            # Apply temperature for scaling the logits
            logits = logits / temperature

            # Logit clipping to avoid extreme values
            logits = torch.clamp(logits, min=-logit_clip_value, max=logit_clip_value)

            # Apply log_softmax for numerical stability
            log_probs = F.log_softmax(logits, dim=-1)

            # Ensure that probabilities are valid
            log_probs = torch.where(torch.isnan(log_probs), torch.full_like(log_probs, -float('Inf')), log_probs)
            log_probs = torch.where(torch.isinf(log_probs), torch.full_like(log_probs, -float('Inf')), log_probs)

            # Convert log probabilities back to normal probabilities
            probs = torch.exp(log_probs)

            # Check if there are NaNs or Infs in probabilities
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                print(f"NaN or Inf detected in probabilities. Resetting to uniform distribution.")
                probs = torch.full_like(probs, 1.0 / probs.numel())  # Reset to uniform distribution if invalid

            # Normalize the probabilities
            probs = probs / probs.sum()

            # Sample the next token
            next_token = torch.multinomial(probs, 1)

            # Append the predicted token to the generated sequence
            generated_sequence = torch.cat([generated_sequence, next_token], dim=1)

            # Stop if the model generates the end-of-sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break

    # Decode the generated sequence to text
    return tokenizer.decode(generated_sequence[0], skip_special_tokens=True)

# === Streamlit Interface ===
st.title("Chatbot with Non-Causal Transformer")

# Input from the user
human_input = st.text_input("Enter Your Input:", "Hello! How are you?")

if st.button('Generate Response'):
    if human_input:
        # Generate Response
        predicted_response = generate_response(human_input, tokenizer, model, device, max_len=50, temperature=1.2)
        
        # Display the response
        st.write(f"Generated Response: {predicted_response}")
    else:
        st.write("Please enter some input.")
