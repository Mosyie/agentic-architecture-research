from abc import ABC, abstractmethod


class ToolEnvironment(ABC):

    @abstractmethod
    def execute_tool(self, tool_name: str, arguments: dict) -> dict:
        pass