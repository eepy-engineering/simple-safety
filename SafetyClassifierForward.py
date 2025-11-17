from torch import load, softmax 
from SafetyClassifier import SafetyClassifier
from transformers import AutoTokenizer
# akhil.bdu.st/simple-safety, 8098 
class SafetyClassifierForward:
    def __init__(self, base_model_name): 
        self.model = SafetyClassifier(base_model_name)

        # expects mounted directory called "data"
        checkpoint_path = 'data/Qwen3-0.6B_safety/'
        self.model.load_state_dict(load(
            f"{checkpoint_path}/simple_safety_cpu.bin",
            map_location='cpu' 
        ))
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None: 
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def eval_text(self, text_input): 
        inputs = self.tokenizer(
            text_input, 
            return_tensors = 'pt', 
            truncation=True, 
            padding=True
        )
        outputs = self.model(**inputs)
        torch_probs = softmax(outputs['logits'], dim=-1)
        prob_unsafe = float(torch_probs[0][0])
        return prob_unsafe
