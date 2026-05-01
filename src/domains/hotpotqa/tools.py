from src.core.tool_environment import ToolEnvironment


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

    def tools_schema(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_chunks",
                    "description": "List available context chunks for the current question.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_content_chunk",
                    "description": "Read one context chunk by chunk_id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chunk_id": {
                                "type": "integer",
                                "description": "The chunk ID to read."
                            }
                        },
                        "required": ["chunk_id"]
                    }
                }
            }
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