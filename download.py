# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import WhisperForConditionalGeneration, WhisperConfig, AutoProcessor, WhisperProcessor, WhisperTokenizer
from peft import PeftModel, PeftConfig
from utils import custom_save


def download_model():
    # do a dry run of loading the huggingface model, which will download weights & config objects
    
    # processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
    # config = WhisperConfig.from_pretrained("openai/whisper-base")
    # tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v2", language="Urdu", task="transcribe")
    # processor = WhisperProcessor.from_pretrained("openai/whisper-large-v2", language="Urdu", task="transcribe")
    # model = WhisperForConditionalGeneration.from_pretrained("Moaiz/whisper-fine-tune-LoRA-roman-urdu-add-2")

    peft_model_id = "Moaiz/whisper-fine-tune-LoRA-roman-urdu-add-2" # Use the same model ID as before.
    language = "Urdu"
    task = "transcribe"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(peft_config.base_model_name_or_path)
    model = PeftModel.from_pretrained(model, peft_model_id, token = "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")
    tokenizer = WhisperTokenizer.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(peft_config.base_model_name_or_path, language=language, task=task)
    custom_save(model)

if __name__ == "__main__":
    download_model()
