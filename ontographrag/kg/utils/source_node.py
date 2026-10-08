from datetime import datetime


class sourceNode:
    """
    Source node class for document metadata
    """

    file_name: str | None = None
    file_size: int | None = None
    file_type: str | None = None
    file_source: str | None = None
    status: str | None = None
    url: str | None = None
    gcsBucket: str | None = None
    gcsBucketFolder: str | None = None
    gcsProjectId: str | None = None
    awsAccessKeyId: str | None = None
    chunkNodeCount: int | None = None
    chunkRelCount: int | None = None
    entityNodeCount: int | None = None
    entityEntityRelCount: int | None = None
    communityNodeCount: int | None = None
    communityRelCount: int | None = None
    node_count: int | None = None
    relationship_count: str | None = None
    model: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    processing_time: float | None = None
    error_message: str | None = None
    total_chunks: int | None = None
    language: str | None = None
    is_cancelled: bool | None = None
    processed_chunk: int | None = None
    access_token: str | None = None
    retry_condition: str | None = None
