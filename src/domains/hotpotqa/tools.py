from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


class ListChunksArgs(BaseModel):
    """No arguments."""


class ReadContentChunkArgs(BaseModel):
    chunk_id: int = Field(description="The chunk ID to read.")


def make_chunks(context: dict) -> list[dict]:
    chunks = []

    for i, (title, sentences) in enumerate(
        zip(context["title"], context["sentences"])
    ):
        chunks.append(
            {
                "chunk_id": i,
                "title": title,
                "text": " ".join(sentences),
            }
        )

    return chunks


def make_hotpotqa_tools(context: dict) -> list[StructuredTool]:
    chunks = make_chunks(context)

    def list_chunks() -> list[dict]:
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"],
            }
            for chunk in chunks
        ]

    def read_content_chunk(chunk_id: int) -> dict:
        if not isinstance(chunk_id, int):
            return {"error": "chunk_id must be an integer"}

        if chunk_id < 0 or chunk_id >= len(chunks):
            return {"error": f"Chunk {chunk_id} not found"}

        return chunks[chunk_id]

    return [
        StructuredTool.from_function(
            func=list_chunks,
            name="list_chunks",
            description="List available context chunks for the current question.",
            args_schema=ListChunksArgs,
        ),
        StructuredTool.from_function(
            func=read_content_chunk,
            name="read_content_chunk",
            description="Read one context chunk by chunk_id.",
            args_schema=ReadContentChunkArgs,
        ),
    ]