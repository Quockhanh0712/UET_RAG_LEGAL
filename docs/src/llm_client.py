# src/llm_client.py
import os
from google import genai
import types
import logging

logger = logging.getLogger(__name__)

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# HuggingFace Transformers Model (Vietnamese Legal LLM)
HF_MODEL = os.getenv("HF_MODEL", "VLSP2025-LegalSML/qwen3-4b-legal-pretrain")
HF_QUANTIZATION = os.getenv("HF_QUANTIZATION", "4bit")  # "none", "4bit", "8bit"
HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "512"))

# Cache for HuggingFace model (lazy load)
_hf_model = None
_hf_tokenizer = None

# Initialize Gemini client
if GEMINI_API_KEY:
    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    # Dummy client for testing without API key
    class DummyClient:
        class DummyModels:
            def generate_content(self, model, contents):
                return types.SimpleNamespace(text="DUMMY_ANSWER_NO_KEY")
        @property
        def models(self):
            return self.DummyModels()
    _gemini_client = DummyClient()


def _load_hf_model():
    """Load HuggingFace model with optional quantization for VRAM efficiency
    
    This is a singleton that loads the model ONCE and caches it globally.
    Even when Chainlit reloads, the model stays in GPU memory.
    """
    global _hf_model, _hf_tokenizer
    
    if _hf_model is not None:
        logger.info("[HF] Using cached model (already loaded)")
        return _hf_model, _hf_tokenizer
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
        
        logger.info(f"[HF] Loading model: {HF_MODEL}")
        logger.info(f"[HF] Quantization: {HF_QUANTIZATION}")
        
        # Load tokenizer
        _hf_tokenizer = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
        
        # Quantization config for saving VRAM
        if HF_QUANTIZATION == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            _hf_model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("[HF] Loaded model with 4-bit quantization")
            
        elif HF_QUANTIZATION == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            _hf_model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("[HF] Loaded model with 8-bit quantization")
            
        else:  # No quantization
            _hf_model = AutoModelForCausalLM.from_pretrained(
                HF_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("[HF] Loaded model without quantization (FP16)")
        
        logger.info(f"[HF] Model loaded successfully!")
        return _hf_model, _hf_tokenizer
        
    except ImportError as e:
        logger.error(f"[HF] Missing package: {e}")
        logger.error("[HF] Run: pip install transformers accelerate bitsandbytes")
        raise
    except Exception as e:
        logger.error(f"[HF] Error loading model: {e}")
        raise


def _generate_with_hf(prompt: str) -> str:
    """Generate answer using HuggingFace Transformers model (Qwen3-4B Legal)"""
    try:
        import torch
        
        model, tokenizer = _load_hf_model()
        
        # Format prompt for Qwen3 - WITH system message forcing Vietnamese
        messages = [
            {"role": "system", "content": "Bạn là trợ lý pháp luật Việt Nam. Bạn PHẢI trả lời BẰNG TIẾNG VIỆT. KHÔNG được trả lời bằng tiếng Anh."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template'):
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        
        # Generate - optimized for speed
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # Reduced for faster generation
                do_sample=True,
                temperature=0.3,  # Lower for more focused answers
                top_p=0.85,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1  # Greedy decoding for speed
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Filter out <think> tags from Qwen3 thinking mode
        import re
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        response = response.strip()
        
        logger.info(f"[HF] Generated {len(response)} characters")
        return response
        
    except Exception as e:
        logger.error(f"[HF] Error: {e}")
        return f"Error generating with HuggingFace model: {str(e)}"

def _generate_with_ollama(prompt: str) -> str:
    """Generate answer using Ollama local LLM"""
    try:
        import ollama
        logger.info(f"[OLLAMA] Generating with model: {OLLAMA_MODEL}")
        
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            options={
                'num_ctx': 2048,  # Reduced for speed
                'temperature': 0.7,  # Higher for faster generation
                'top_p': 0.9,
            }
        )
        
        answer = response['message']['content']
        logger.info(f"[OLLAMA] Generated {len(answer)} characters")
        return answer
        
    except ImportError:
        logger.error("[OLLAMA] ollama package not installed. Run: pip install ollama")
        return "Error: Ollama Python package not installed. Please run: pip install ollama"
    except Exception as e:
        logger.error(f"[OLLAMA] Error: {e}")
        return f"Error calling Ollama: {str(e)}"


def generate_with_ollama_stream(prompt: str):
    """Generate answer using Ollama with streaming (generator)"""
    try:
        import ollama
        logger.info(f"[OLLAMA] Streaming with model: {OLLAMA_MODEL}")
        
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt
            }],
            stream=True,
            options={
                'num_ctx': 2048,  # Reduced for speed
                'temperature': 0.7,  # Higher for faster generation
                'top_p': 0.9,
            }
        )
        
        for chunk in stream:
            content = chunk['message']['content']
            if content:
                yield content
                
    except Exception as e:
        logger.error(f"[OLLAMA] Stream error: {e}")
        yield f"Lỗi: {str(e)}"

def _generate_with_gemini(prompt: str) -> str:
    """Generate answer using Google Gemini API"""
    logger.info(f"[GEMINI] Generating with model: {GEMINI_MODEL}")
    resp = _gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )
    logger.info(f"[GEMINI] Generated {len(resp.text)} characters")
    return resp.text

def generate_answer(prompt: str) -> str:
    """
    Generate answer using configured LLM provider.
    
    Provider is selected based on LLM_PROVIDER environment variable:
    - "ollama": Uses local Ollama (GPU, free, offline)
    - "gemini": Uses Google Gemini API (cloud, requires API key)
    - "transformers" or "hf": Uses HuggingFace Transformers (local GPU, Qwen3-4B Legal)
    """
    logger.info(f"[LLM] Using provider: {LLM_PROVIDER}")
    
    if LLM_PROVIDER == "ollama":
        return _generate_with_ollama(prompt)
    elif LLM_PROVIDER in ("transformers", "hf", "huggingface"):
        return _generate_with_hf(prompt)
    else:  # default to gemini
        return _generate_with_gemini(prompt)


def get_available_providers() -> list:
    """Return list of available LLM providers"""
    providers = ["gemini"]  # Always available (with API key)
    
    # Check Ollama
    try:
        import ollama
        providers.append("ollama")
    except ImportError:
        pass
    
    # Check Transformers
    try:
        import transformers
        providers.append("transformers")
    except ImportError:
        pass
    
    return providers

