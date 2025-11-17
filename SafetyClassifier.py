from transformers import AutoModelForCausalLM
from torch import float32 as torchfloat32 
import torch.nn as nn

class SafetyClassifier(nn.Module):
    def __init__(self, base_model_name, num_labels=2):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(base_model_name, 
            trust_remote_code=True, 
            torch_dtype=torchfloat32
        )
        hidden_size = self.model.config.hidden_size

        # Attach a classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        ).to(dtype=torchfloat32)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )
        # Use the hidden state of the last token (like GPT2 classification head)
        last_hidden_state = outputs.hidden_states[-1]
        last_token = last_hidden_state[:, -1, :]   # (batch, hidden)
        logits = self.classifier(last_token)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return {"loss": loss, "logits": logits}