import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    NotRequired,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    Unpack,
    cast,
)

import google.generativeai as _genai
import httpx
from google.generativeai.types.content_types import (
    ContentsType,
    FunctionLibraryType,
    ToolConfigType,
)
from google.generativeai.types.generation_types import (
    GenerationConfig,
    GenerationConfigType,
)
from google.generativeai.types.helper_types import RequestOptionsType
from google.generativeai.types.safety_types import SafetySettingOptions
from huggingface_hub import (
    InferenceClient,
    TextGenerationInputGrammarType,
    TextGenerationOutput,
    TextGenerationStreamOutput,
)
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableConfig
from openai import NotGiven, Omit, Stream
from openai import OpenAI as OpenAIClient
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAudioParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from openai.types.shared import ReasoningEffort
from openai.types.shared_params import Metadata

if TYPE_CHECKING:
    import ollama
    from langchain_community.embeddings import (
        HuggingFaceEmbeddings as LegacyEmbeddingType,
    )
    from langchain_huggingface import HuggingFaceEmbeddings as ModernEmbeddingType

from langchain_core.runnables.base import Runnable

from ontographrag.kg.utils.common_functions import _resolve_huggingface_embedding_model

logger: logging.Logger = logging.getLogger(__name__)

ModelOutput: TypeAlias = (
    str
    | None
    | TextGenerationOutput
    | Iterable[str]
    | Iterable[TextGenerationStreamOutput]
)


class CompletionMessage(TypedDict):
    content: str | None


class CompletionChoice(TypedDict):
    message: CompletionMessage


class CompletionPayload(TypedDict):
    choices: list[CompletionChoice]


# LangChain BaseMessage.content deliberately supports multimodal content, which
# the existing adapter forwards unchanged rather than flattening it into text.
MessageContent: TypeAlias = str | list[str | dict[str, Any]]


# This is the JSON encoder's value boundary, not a generic SDK-object map.
JSONValue: TypeAlias = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)


class OpenAIOptions(TypedDict, total=False):
    """openai 1.98.0 Completions.create kwargs excluding owned model/messages."""

    audio: ChatCompletionAudioParam | None | NotGiven
    frequency_penalty: float | None | NotGiven
    function_call: completion_create_params.FunctionCall | NotGiven
    functions: Iterable[completion_create_params.Function] | NotGiven
    logit_bias: dict[str, int] | None | NotGiven
    logprobs: bool | None | NotGiven
    max_completion_tokens: int | None | NotGiven
    max_tokens: int | None | NotGiven
    metadata: Metadata | None | NotGiven
    modalities: list[Literal["text", "audio"]] | None | NotGiven
    n: int | None | NotGiven
    parallel_tool_calls: bool | NotGiven
    prediction: ChatCompletionPredictionContentParam | None | NotGiven
    presence_penalty: float | None | NotGiven
    prompt_cache_key: str | NotGiven
    reasoning_effort: ReasoningEffort | None | NotGiven
    response_format: completion_create_params.ResponseFormat | NotGiven
    safety_identifier: str | NotGiven
    seed: int | None | NotGiven
    service_tier: (
        Literal["auto", "default", "flex", "scale", "priority"] | None | NotGiven
    )
    stop: str | list[str] | None | NotGiven
    store: bool | None | NotGiven
    stream: bool | None | NotGiven
    stream_options: ChatCompletionStreamOptionsParam | None | NotGiven
    temperature: float | None | NotGiven
    tool_choice: ChatCompletionToolChoiceOptionParam | NotGiven
    tools: Iterable[ChatCompletionToolParam] | NotGiven
    top_logprobs: int | None | NotGiven
    top_p: float | None | NotGiven
    user: str | NotGiven
    web_search_options: completion_create_params.WebSearchOptions | NotGiven
    extra_headers: Mapping[str, str | Omit] | None
    extra_query: Mapping[str, object] | None
    extra_body: object
    timeout: float | httpx.Timeout | None | NotGiven


class HuggingFaceOptions(TypedDict, total=False):
    """InferenceClient.text_generation options, including stream/details."""

    details: bool
    stream: bool
    adapter_id: str | None
    best_of: int | None
    decoder_input_details: bool | None
    do_sample: bool | None
    frequency_penalty: float | None
    grammar: TextGenerationInputGrammarType | None
    max_new_tokens: int | None
    repetition_penalty: float | None
    return_full_text: bool | None
    seed: int | None
    stop: list[str] | None
    stop_sequences: list[str] | None
    temperature: float | None
    top_k: int | None
    top_n_tokens: int | None
    top_p: float | None
    truncate: int | None
    typical_p: float | None
    watermark: bool | None


class GeminiGenerationOptions(TypedDict, total=False):
    generation_config: GenerationConfigType | None
    safety_settings: SafetySettingOptions | None
    stream: bool
    tools: FunctionLibraryType | None
    tool_config: ToolConfigType | None
    request_options: RequestOptionsType | None


class GeminiOptions(GeminiGenerationOptions, total=False):
    # This value is discarded, not numerically inspected or sent to the SDK.
    temperature: object


class GeminiResponse(Protocol):
    @property
    def text(self) -> str: ...


class GeminiModel(Protocol):
    def generate_content(
        self,
        contents: ContentsType,
        **kwargs: Unpack[GeminiGenerationOptions],
    ) -> GeminiResponse: ...


class GeminiModelFactory(Protocol):
    def __call__(
        self,
        model_name: str,
        *,
        generation_config: GenerationConfigType | None = None,
    ) -> GeminiModel: ...


class GeminiConfigure(Protocol):
    def __call__(self, *, api_key: str | None = None) -> None: ...


class GeminiModule(Protocol):
    # __init__.py explicitly reexports these installed implementation symbols.
    configure: GeminiConfigure
    GenerativeModel: GeminiModelFactory
    GenerationConfig: type[GenerationConfig]


genai: GeminiModule = cast(GeminiModule, _genai)


class OllamaChatOptions(TypedDict, total=False):
    messages: "Sequence[Mapping[str, Any] | ollama.Message] | None"
    tools: "Sequence[Mapping[str, Any] | ollama.Tool | Callable[..., object]] | None"
    stream: bool
    format: Literal["", "json"] | dict[str, Any] | None
    options: "Mapping[str, Any] | ollama.Options | None"
    keep_alive: float | str | None


class OllamaOptions(OllamaChatOptions, total=False):
    # Existing behavior tests truthiness rather than requiring a boolean.
    force_json: object


class OllamaChatRequest(OllamaChatOptions, total=False):
    model: str


class OpenAICompletion(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: Iterable[ChatCompletionMessageParam],
        **kwargs: Unpack[OpenAIOptions],
    ) -> ChatCompletion | Stream[ChatCompletionChunk]: ...


class OllamaChat(Protocol):
    def __call__(
        self,
        **kwargs: Unpack[OllamaChatRequest],
    ) -> "ollama.ChatResponse | Iterable[ollama.ChatResponse]": ...


class HuggingFaceGeneration(Protocol):
    def __call__(
        self,
        prompt: str,
        *,
        model: str | None = None,
        **kwargs: Unpack[HuggingFaceOptions],
    ) -> (
        str
        | TextGenerationOutput
        | Iterable[str]
        | Iterable[TextGenerationStreamOutput]
    ): ...


class ProviderMessage(TypedDict):
    role: Literal["system", "user"]
    content: MessageContent


HTTPPayloadValue: TypeAlias = JSONValue | list[ProviderMessage]


class AdapterInput(TypedDict):
    """Well-formed caller inputs, not a schema inferred by the dict guard."""

    messages: NotRequired[Iterable[object]]
    model: NotRequired[str]
    input: NotRequired[MessageContent]
    text: NotRequired[MessageContent]
    chunk_text: NotRequired[MessageContent]


# These describe known producers. The attribute-existence checks in invoke do
# not establish these contracts for arbitrary public duck inputs.
class ContentMessage(Protocol):
    @property
    def content(self) -> MessageContent: ...


class RoleMessage(ContentMessage, Protocol):
    @property
    def type(self) -> str: ...


class PromptMessages(Protocol):
    def to_messages(self) -> Iterable[object]: ...


class ModelProvider(ABC):
    # Extension providers own their kwargs schema; finite built-in SDK options
    # below intentionally expose their narrower contracts.
    @abstractmethod
    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: Any,
    ) -> ModelOutput:
        pass


class LangChainRunnableAdapter(Runnable[object, ModelOutput]):
    def __init__(
        self,
        provider: ModelProvider,
        model: str,
        temperature: float = 1.0,
    ) -> None:
        self.provider: ModelProvider = provider
        self.model: str = model
        self.temperature: float = temperature

    def invoke(
        self,
        input: object,
        config: RunnableConfig | None = None,
    ) -> ModelOutput:
        # Handle ChatPromptTemplate inputs (system and user messages)
        if isinstance(input, dict):
            # Direct dict input - extract model name if provided
            input_dict: dict[object, object] = input
            messages: object = input_dict.get("messages", [])
            model_name: object = input_dict.get("model", self.model)
        elif hasattr(input, "to_messages"):
            # ChatPromptValue object
            messages = input.to_messages()
            model_name = self.model
        else:
            # Fallback
            messages = []
            model_name = self.model

        # Public duck inputs are not validated; preserve raw values and failures.
        system_prompt: object = ""
        user_prompt: object = ""

        message: object
        for message in messages:
            if hasattr(message, "type") and hasattr(message, "content"):
                if message.type == "system":
                    system_prompt = message.content
                elif message.type == "human":
                    user_prompt = message.content

        # If no explicit messages found, try simple string input
        if not system_prompt and not user_prompt:
            if isinstance(input, dict):
                fallback_input: dict[object, object] = input
                user_prompt = fallback_input.get(
                    "input",
                    fallback_input.get("text", fallback_input.get("chunk_text", "")),
                )
            elif hasattr(input, "content"):
                user_prompt = input.content
            elif isinstance(input, str):
                user_prompt = input
            else:
                user_prompt = str(input)

        return self.provider.generate(
            system_prompt,
            user_prompt,
            model_name,
            temperature=self.temperature,
        )

    def __class_getitem__(cls, item: object) -> type[Self]:
        return cls

    def with_structured_output(self, schema: object, **kwargs: object) -> Self:
        """Add structured output support for LLMGraphTransformer compatibility"""
        return self

    def bind(self, **kwargs: object) -> Self:
        """Add bind method for LangChain compatibility"""
        return self


LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


class OpenAIProvider(ModelProvider):
    def __init__(self) -> None:
        self.client: OpenAIClient = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY"),
            timeout=LLM_TIMEOUT_SECONDS,
        )

    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: Unpack[OpenAIOptions],
    ) -> str | None:
        kwargs.setdefault("temperature", 1.0)
        # Raw LangChain blocks are forwarded, including SDK-incompatible shapes.
        # Do not cast their contents into validated OpenAI message parameters.
        create: OpenAICompletion = cast(
            OpenAICompletion,
            self.client.chat.completions.create,
        )
        response: ChatCompletion | Stream[ChatCompletionChunk] = create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content


class OllamaProvider(ModelProvider):
    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: Unpack[OllamaOptions],
    ) -> str:
        import ollama

        try:
            # Check if JSON format is explicitly requested in kwargs
            force_json: object = kwargs.pop("force_json", False)

            # Build the chat request
            chat_kwargs: OllamaChatRequest = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"num_ctx": 16384},
            }

            # Only enforce JSON format if explicitly requested
            if force_json:
                chat_kwargs["format"] = "json"

            # Add any remaining kwargs
            remaining_options: OllamaChatRequest = cast(OllamaChatRequest, kwargs)
            chat_kwargs.update(remaining_options)

            chat: OllamaChat = cast(OllamaChat, ollama.chat)
            response: ollama.ChatResponse | Iterable[ollama.ChatResponse] = chat(
                **chat_kwargs,
            )
            content: str = response["message"]["content"]

            # If JSON was requested, validate the response
            if force_json:
                try:
                    json.loads(content)
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON response from model: {content}")

            return content
        except Exception as e:
            if "not found" in str(e).lower():
                return f"Error: Model '{model}' not found. Please run 'ollama pull {model}' to download it."
            else:
                return f"Ollama error: {e!s}"


class GeminiProvider(ModelProvider):
    def __init__(self) -> None:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: Unpack[GeminiOptions],
    ) -> str:
        kwargs.pop("temperature", None)  # pass via generation_config instead
        gemini_model: GeminiModel = genai.GenerativeModel(
            model,
            generation_config=genai.GenerationConfig(temperature=0),
        )
        generation_options: GeminiGenerationOptions = kwargs
        response: GeminiResponse = gemini_model.generate_content(
            contents=[system_prompt, user_prompt],
            **generation_options,
        )
        # The SDK accessor is unannotated but returns a string join or raises.
        content: str = response.text
        return content


class HuggingFaceProvider(ModelProvider):
    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: Unpack[HuggingFaceOptions],
    ) -> (
        str
        | TextGenerationOutput
        | Iterable[str]
        | Iterable[TextGenerationStreamOutput]
    ):
        client: InferenceClient = InferenceClient(token=os.getenv("HF_API_TOKEN"))
        full_prompt: str = f"{system_prompt}\n\n{user_prompt}"
        kwargs.setdefault("temperature", 0.0)
        generate: HuggingFaceGeneration = cast(
            HuggingFaceGeneration,
            client.text_generation,
        )
        response: (
            str
            | TextGenerationOutput
            | Iterable[str]
            | Iterable[TextGenerationStreamOutput]
        ) = generate(prompt=full_prompt, model=model, **kwargs)
        return response


class DeepSeekProvider(ModelProvider):
    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: JSONValue,
    ) -> str | None:
        api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
        headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
        kwargs.setdefault("temperature", 0)
        messages: list[ProviderMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: dict[str, HTTPPayloadValue] = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        response: httpx.Response = httpx.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        completion: CompletionPayload = response.json()
        return completion["choices"][0]["message"]["content"]


class OpenRouterProvider(ModelProvider):
    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: JSONValue,
    ) -> str | None:
        api_key: str | None = os.getenv("OPENROUTER_API_KEY")
        headers: dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/ModelContext/tool-2025-kg-rag",
            "X-Title": "KG RAG Tool",
        }
        kwargs.setdefault("temperature", 0)
        messages: list[ProviderMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload: dict[str, HTTPPayloadValue] = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        response: httpx.Response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        completion: CompletionPayload = response.json()
        return completion["choices"][0]["message"]["content"]

    def with_structured_output(self, schema: object, **kwargs: object) -> Self:
        """Add structured output support for LLMGraphTransformer compatibility"""
        return self

    def bind(self, **kwargs: object) -> Self:
        """Add bind method for LangChain compatibility"""
        return self


class TemperatureLockedProvider(ModelProvider):
    """Wrap a provider and force a fixed temperature on every generate call."""

    def __init__(self, provider: ModelProvider, temperature: float = 0.0) -> None:
        self.provider: ModelProvider = provider
        self.temperature: float = temperature

    def generate(
        self,
        system_prompt: MessageContent,
        user_prompt: MessageContent,
        model: str,
        **kwargs: Any,
    ) -> ModelOutput:
        kwargs = dict(kwargs)
        kwargs["temperature"] = self.temperature
        return self.provider.generate(system_prompt, user_prompt, model, **kwargs)

    # Attribute names and types are supplied by arbitrary extension providers.
    def __getattr__(self, name: str) -> Any:
        return getattr(self.provider, name)


def get_provider(
    provider: str,
    model: str | None = None,
    **kwargs: object,
) -> ModelProvider:
    providers: dict[str, type[ModelProvider]] = {
        "openai": OpenAIProvider,
        "ollama": OllamaProvider,
        "gemini": GeminiProvider,
        "huggingface": HuggingFaceProvider,
        "deepseek": DeepSeekProvider,
        "openrouter": OpenRouterProvider,
    }

    provider_class: type[ModelProvider] | None = providers.get(provider.lower())
    if not provider_class:
        raise ValueError(f"Unsupported provider: {provider}")

    return provider_class()


class ProviderFactory(Protocol):
    def __call__(
        self,
        provider: str,
        model: str | None = None,
        **kwargs: object,
    ) -> ModelProvider: ...


# alias for compatibility
get_llm_provider: ProviderFactory = get_provider


def get_embedding_model(provider: str = "huggingface") -> "Embeddings":
    """Get an embedding model from different providers

    Args:
        provider: Provider to use ('huggingface', 'openai', 'vertexai')

    Returns:
        LangChain embedding model
    """
    if provider == "huggingface":
        try:
            # Try to import from the correct package first
            embedding_class: type[ModernEmbeddingType | LegacyEmbeddingType]
            try:
                from langchain_huggingface import HuggingFaceEmbeddings

                embedding_class = HuggingFaceEmbeddings
            except ImportError:
                try:
                    from langchain_community.embeddings import (
                        HuggingFaceEmbeddings as CommunityHuggingFaceEmbeddings,
                    )

                    embedding_class = CommunityHuggingFaceEmbeddings
                except ImportError:
                    logger.error("No compatible HuggingFace embeddings package found")
                    raise

            logger.info("Importing HuggingFaceEmbeddings")
            model_name: str = _resolve_huggingface_embedding_model()
            embedder: Embeddings = embedding_class(
                model_name=model_name,
                model_kwargs={"device": "cpu"},  # Use CPU by default for compatibility
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("HuggingFaceEmbeddings initialized successfully")
            return embedder
        except Exception as e:
            logger.warning("HuggingFace initialization failed: %s", e)
            raise ImportError(
                "huggingface embeddings not available. Install with: pip install sentence-transformers transformers torch",
            )
    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        )
    elif provider == "vertexai":
        from langchain_google_vertexai import VertexAIEmbeddings

        return VertexAIEmbeddings(
            model=os.getenv("VERTEXAI_EMBEDDING_MODEL", "text-embedding-005").strip(),
        )
    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. Choose from 'huggingface', 'openai', 'vertexai'",
        )


def get_embedding_method(
    provider_name: str | None = None,
) -> tuple[str, "Embeddings"]:
    """Get the configured embedding method from environment

    Args:
        provider_name: Optional provider override ('huggingface', 'openai', 'vertexai')

    Returns:
        tuple: (provider_name, embedding_model)
    """
    if provider_name is None:
        provider: str = os.getenv(
            "EMBEDDING_PROVIDER",
            "huggingface",
        )  # Default to huggingface
    else:
        provider = provider_name

    try:
        embedder: Embeddings = get_embedding_model(provider)
        return provider, embedder
    except Exception as e:
        logger.warning(
            "Failed to initialize %s embeddings; falling back to OpenAI: %s",
            provider,
            e,
        )
        # Fallback to OpenAI if HuggingFace fails
        try:
            embedder = get_embedding_model("openai")
            return "openai", embedder
        except Exception as e2:
            raise RuntimeError(
                f"Failed to initialize any embedding model. OpenAI error: {e2}",
            )
