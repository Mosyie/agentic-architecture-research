from abc import ABC, abstractmethod


class ToolEnvironment(ABC):

    @abstractmethod
    def tools_schema(self) -> list[dict]:
        pass

    @abstractmethod
    def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        pass