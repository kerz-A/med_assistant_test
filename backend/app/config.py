from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # LLM
    llm_provider: str = "gigachat"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen2.5:3b"
    openrouter_api_key: str = ""
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    gigachat_auth_key: str = ""
    gigachat_model: str = "GigaChat-2-Max"

    # Whisper
    whisper_model: str = "medium"
    whisper_device: str = "auto"
    whisper_compute_type: str = "float16"
    whisper_beam_size: int = 3

    # Processing
    processing_interval_seconds: int = 8
    audio_buffer_duration_seconds: int = 120
    audio_overlap_seconds: float = 2.0
    num_speakers: int = 2

    # VAD (Silero)
    vad_threshold: float = 0.5
    vad_silence_ms: int = 600       # silence duration to end segment
    vad_min_speech_ms: int = 500    # minimum speech segment duration
    vad_max_speech_ms: int = 30000  # max speech segment (force-emit)
    vad_speech_pad_ms: int = 100    # padding around speech

    # LLM extraction batching
    extraction_batch_size: int = 5       # utterances before triggering extraction (was 3)
    extraction_interval_seconds: float = 20.0  # max seconds between extractions (was 15)

    # LLM call pacing
    llm_min_gap_seconds: float = 0.0  # min gap between LLM calls (0=no delay for GigaChat, 2.0 for Groq)
    llm_max_concurrent: int = 1  # max concurrent HTTP requests to LLM (1 for GigaChat free tier)

    # Speaker ID
    speaker_similarity_threshold: float = 0.1  # min diff between doctor/patient cosine sims
    calibration_same_speaker_threshold: float = 0.75  # cosine sim above this = same speaker during calibration
    max_concurrent_segments: int = 0  # 0=auto (1 CPU, 2 GPU), or set manually

    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000

    @property
    def llm_api_key(self) -> str:
        match self.llm_provider:
            case "groq": return self.groq_api_key
            case "openai": return self.openai_api_key
            case "openrouter": return self.openrouter_api_key
            case "deepseek": return self.deepseek_api_key
            case "gigachat": return self.gigachat_auth_key
            case _: return ""

    @property
    def llm_model(self) -> str:
        match self.llm_provider:
            case "groq": return self.groq_model
            case "openai": return self.openai_model
            case "ollama": return self.ollama_model
            case "openrouter": return self.openrouter_model
            case "deepseek": return self.deepseek_model
            case "gigachat": return self.gigachat_model
            case _: return self.groq_model

    @property
    def llm_base_url(self) -> str:
        match self.llm_provider:
            case "groq": return "https://api.groq.com/openai/v1"
            case "openai": return "https://api.openai.com/v1"
            case "ollama": return f"{self.ollama_base_url}/v1"
            case "openrouter": return "https://openrouter.ai/api/v1"
            case "deepseek": return "https://api.deepseek.com/v1"
            case "gigachat": return "https://gigachat.devices.sberbank.ru/api/v1"
            case _: return "https://api.groq.com/openai/v1"


settings = Settings()
