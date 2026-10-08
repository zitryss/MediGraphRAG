import hashlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired

import boto3
from dotenv import load_dotenv
from langchain_community.embeddings import BedrockEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jGraph
from langchain_openai import OpenAIEmbeddings
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from _hashlib import HASH

    from mypy_boto3_bedrock_runtime import BedrockRuntimeClient


class OpenAIEmbeddingConfig(TypedDict):
    model: str
    dimensions: NotRequired[int]


class VertexEmbeddingConfig(TypedDict):
    model: str


# Load environment variables
load_dotenv()

_OPENAI_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_VERTEXAI_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-005": 768,
    "text-multilingual-embedding-002": 768,
    "gemini-embedding-001": 3072,
}

_HF_EMBEDDING_DEFAULTS_BY_PROFILE: dict[str, str] = {
    "balanced": "sentence-transformers/all-MiniLM-L6-v2",
    "accuracy": "BAAI/bge-base-en-v1.5",
}


def _resolve_openai_embedding_config() -> tuple[OpenAIEmbeddingConfig, int]:
    model_name: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    ).strip()
    dimensions_env: str = os.getenv("OPENAI_EMBEDDING_DIMENSION", "").strip()
    dimension: int | None
    kwargs: OpenAIEmbeddingConfig = {"model": model_name}
    if dimensions_env:
        dimension = int(dimensions_env)
        kwargs["dimensions"] = dimension
        return kwargs, dimension

    dimension = _OPENAI_EMBEDDING_DIMENSIONS.get(model_name)
    if dimension is None:
        logging.warning(
            "Unknown OpenAI embedding model '%s'; defaulting dimension to 1536. "
            "Set OPENAI_EMBEDDING_DIMENSION explicitly if needed.",
            model_name,
        )
        dimension = 1536
    return kwargs, dimension


def _resolve_vertexai_embedding_config() -> tuple[VertexEmbeddingConfig, int]:
    model_name: str = os.getenv(
        "VERTEXAI_EMBEDDING_MODEL",
        "text-embedding-005",
    ).strip()
    dimensions_env: str = os.getenv("VERTEXAI_EMBEDDING_DIMENSION", "").strip()
    kwargs: VertexEmbeddingConfig = {"model": model_name}
    if dimensions_env:
        return kwargs, int(dimensions_env)

    dimension: int | None = _VERTEXAI_EMBEDDING_DIMENSIONS.get(model_name)
    if dimension is None:
        logging.warning(
            "Unknown Vertex AI embedding model '%s'; defaulting dimension to 768. "
            "Set VERTEXAI_EMBEDDING_DIMENSION explicitly if needed.",
            model_name,
        )
        dimension = 768
    return kwargs, dimension


def _resolve_huggingface_embedding_model(explicit_model_name: str = "") -> str:
    normalized: str = str(explicit_model_name or "").strip()
    normalized_lower: str = normalized.lower()
    if normalized and normalized_lower not in {"sentence_transformers", "huggingface"}:
        return normalized

    env_model: str = os.getenv("HUGGINGFACE_EMBEDDING_MODEL", "").strip()
    if env_model:
        return env_model

    profile: str = (
        str(os.getenv("ONTOGRAPHRAG_RETRIEVAL_PROFILE", "balanced")).strip().lower()
    )
    if profile in {"accuracy", "quality", "high_accuracy"}:
        return _HF_EMBEDDING_DEFAULTS_BY_PROFILE["accuracy"]
    return _HF_EMBEDDING_DEFAULTS_BY_PROFILE["balanced"]


def load_embedding_model(embedding_model_name: str | None) -> tuple[Embeddings, int]:
    """
    Load embedding model based on the model name
    Returns embeddings object and dimension
    """
    embeddings: Embeddings
    dimension: int
    normalized: str = str(embedding_model_name or "").strip()
    normalized_lower: str = normalized.lower()

    if normalized_lower in {"openai"} or normalized in _OPENAI_EMBEDDING_DIMENSIONS:
        kwargs, dimension = _resolve_openai_embedding_config()
        if normalized in _OPENAI_EMBEDDING_DIMENSIONS:
            kwargs["model"] = normalized
            dimension = _OPENAI_EMBEDDING_DIMENSIONS[normalized]
        embeddings = OpenAIEmbeddings(**kwargs)
        logging.info(
            "Embedding: Using OpenAI Embeddings model=%s, Dimension:%s",
            kwargs["model"],
            dimension,
        )
    elif (
        normalized_lower in {"vertexai"} or normalized in _VERTEXAI_EMBEDDING_DIMENSIONS
    ):
        vertex_kwargs, dimension = _resolve_vertexai_embedding_config()
        if normalized in _VERTEXAI_EMBEDDING_DIMENSIONS:
            vertex_kwargs["model"] = normalized
            dimension = _VERTEXAI_EMBEDDING_DIMENSIONS[normalized]
        embeddings = VertexAIEmbeddings(**vertex_kwargs)
        logging.info(
            "Embedding: Using Vertex AI Embeddings model=%s, Dimension:%s",
            vertex_kwargs["model"],
            dimension,
        )
    elif normalized_lower == "titan":
        embeddings = get_bedrock_embeddings()
        dimension = 1536
        logging.info(
            f"Embedding: Using bedrock titan Embeddings , Dimension:{dimension}",
        )
    else:
        model_name: str = _resolve_huggingface_embedding_model(normalized)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,  # , cache_folder="/embedding_model"
        )
        sample_vector: list[float] = embeddings.embed_query("embedding readiness probe")
        dimension = len(sample_vector)
        logging.info(
            "Embedding: Using Langchain HuggingFaceEmbeddings model=%s, Dimension:%s",
            model_name,
            dimension,
        )
    return embeddings, dimension


def create_graph_database_connection(
    uri: str,
    userName: str,
    password: str,
    database: str,
) -> Neo4jGraph:
    """
    Create Neo4j graph database connection
    """
    enable_user_agent: bool = os.environ.get("ENABLE_USER_AGENT", "False").lower() in (
        "true",
        "1",
        "yes",
    )

    driver_config: dict[str, str | None] = {}
    if enable_user_agent:
        driver_config["user_agent"] = os.environ.get("NEO4J_USER_AGENT")

    # LangChain Neo4jGraph with direct username/password parameters
    graph: Neo4jGraph = Neo4jGraph(
        url=uri,
        database=database,
        username=userName,
        password=password,
        refresh_schema=False,
        sanitize=True,
        driver_config=driver_config,
    )
    return graph


def delete_uploaded_local_file(merged_file_path: str | Path, file_name: str) -> None:
    """
    Delete uploaded local file
    """
    file_path: Path = Path(merged_file_path)
    if file_path.exists():
        file_path.unlink()
        logging.info(f"file {file_name} deleted successfully")


def create_gcs_bucket_folder_name_hashed(uri: str, file_name: str) -> str:
    """
    Create GCS bucket folder name with hash
    """
    folder_name: str = uri + file_name
    folder_name_sha1: HASH = hashlib.sha1(folder_name.encode())
    folder_name_sha1_hashed: str = folder_name_sha1.hexdigest()
    return folder_name_sha1_hashed


def get_bedrock_embeddings() -> BedrockEmbeddings:
    """
    Creates and returns a BedrockEmbeddings object using the specified model name.
    Args:
        model (str): The name of the model to use for embeddings.
    Returns:
        BedrockEmbeddings: An instance of the BedrockEmbeddings class.
    """
    try:
        env_value: str | None = os.getenv("BEDROCK_EMBEDDING_MODEL")
        if not env_value:
            raise ValueError(
                "Environment variable 'BEDROCK_EMBEDDING_MODEL' is not set.",
            )
        try:
            model_name, aws_access_key, aws_secret_key, region_name = env_value.split(
                ",",
            )
        except ValueError:
            raise ValueError(
                "Environment variable 'BEDROCK_EMBEDDING_MODEL' is improperly formatted. "
                "Expected format: 'model_name,aws_access_key,aws_secret_key,region_name'.",
            )
        bedrock_client: BedrockRuntimeClient = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name.strip(),
            aws_access_key_id=aws_access_key.strip(),
            aws_secret_access_key=aws_secret_key.strip(),
        )
        bedrock_embeddings: BedrockEmbeddings = BedrockEmbeddings(
            model_id=model_name.strip(),
            client=bedrock_client,
        )
        return bedrock_embeddings
    except Exception as _e:
        logging.exception("Unexpected error while loading Bedrock embeddings")
        raise
