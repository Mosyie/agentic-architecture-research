from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from src.core.tool_environment import ToolEnvironment


class ListChunksArgs(BaseModel):
    """No arguments."""


class ReadContentChunkArgs(BaseModel):
    chunk_id: int = Field(description="The chunk ID to read.")

class HotpotQAToolEnvironment(ToolEnvironment):

    def __init__(self, context: dict):
        self.chunks = self._make_chunks(context)

    def _make_chunks(self, context: dict) -> list[dict]:
        chunks = []

        for i, (title, sentences) in enumerate(
            zip(context["title"], context["sentences"])
        ):
            chunks.append({
                "chunk_id": i,
                "title": title,
                "text": " ".join(sentences)
            })

        return chunks

    def as_langchain_tools(self) -> list[StructuredTool]:
        """
        Adapt this environment into LangChain StructuredTools.

        The tools route through execute_tool(), so this class remains the
        source of truth for tool behavior.
        """

        def list_chunks() -> list:
            return self.execute_tool("list_chunks", {})

        def read_content_chunk(chunk_id: int) -> dict:
            return self.execute_tool(
                "read_content_chunk",
                {"chunk_id": chunk_id},
            )

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

    def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        if tool_name == "list_chunks":
            return self._list_chunks()

        if tool_name == "read_content_chunk":
            return self._read_content_chunk(arguments.get("chunk_id"))

        return {"error": f"Unknown tool: {tool_name}"}

    def _list_chunks(self) -> list[dict]:
        return [
            {
                "chunk_id": chunk["chunk_id"],
                "title": chunk["title"]
            }
            for chunk in self.chunks
        ]

    def _read_content_chunk(self, chunk_id: int) -> dict:
        if not isinstance(chunk_id, int):
            return {"error": "chunk_id must be an integer"}

        if chunk_id < 0 or chunk_id >= len(self.chunks):
            return {"error": f"Chunk {chunk_id} not found"}

        return self.chunks[chunk_id]
