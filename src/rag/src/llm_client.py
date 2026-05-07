import os
import requests
import json
import re
from typing import Any, List, Mapping, Optional, Dict, Tuple
from dotenv import load_dotenv
from pydantic import PrivateAttr

from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.tools import tool

load_dotenv()


class BielikChatModel(BaseChatModel):
    temperature: float = 0.0

    _api_url: str = PrivateAttr(default="https://153.19.237.85/api/llm/prompt/chat")
    _max_length: int = PrivateAttr(default=32768)
    _auth_kwargs: dict = PrivateAttr(default_factory=lambda: {
        "auth": (os.getenv("LLM_USERNAME"), os.getenv("LLM_PASSWORD")),
        "verify": False
    })

    @property
    def _llm_type(self) -> str:
        return "bielik_chat_model"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"temperature": self.temperature}

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> Any:
        formatted_tools = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted_tools, **kwargs)

    def _format_messages(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            formatted.append({"role": role, "content": msg.content})
        return formatted

    def _inject_tool_instructions(self, messages: List[BaseMessage], tools: Optional[List[Dict]]) -> List[BaseMessage]:
        if not tools:
            return list(messages)

        modified_messages = list(messages)
        tools_str = json.dumps(tools, indent=2, ensure_ascii=False)

        tool_instruction = (
            "\n\n--- TRYB ASYSTENTA Z NARZĘDZIAMI ---\n"
            f"Masz opcjonalny dostęp do następujących funkcji:\n{tools_str}\n\n"
            "KRYTYCZNE ZASADY (PRZECZYTAJ UWAŻNIE):\n"
            "1. ZABRONIONE jest używanie narzędzi, jeśli użytkownik tylko się wita, prowadzi luźną rozmowę lub zadaje pytanie, na które znasz odpowiedź.\n"
            "2. Używaj narzędzi TYLKO I WYŁĄCZNIE wtedy, gdy potrzebujesz zewnętrznych danych do wykonania zadania.\n"
            "3. Jeśli odpowiadasz normalnie, po prostu napisz tekst.\n"
            "4. Jeśli MUSISZ użyć narzędzia, odpowiedz WYŁĄCZNIE tagami <tool_call> i niczym więcej.\n\n"
            "PRZYKŁADY PRAWIDŁOWYCH ZACHOWAŃ:\n"
            "[Przykład 1 - Luźna rozmowa]\n"
            "Użytkownik: Cześć, co tam?\n"
            "Asystent: Cześć! Jestem wirtualnym asystentem. W czym mogę Ci dzisiaj pomóc?\n\n"
            "[Przykład 2 - Zwykła wiedza]\n"
            "Użytkownik: Ile to 2+2?\n"
            "Asystent: 2+2 to 4.\n\n"
            "[Przykład 3 - Konieczność użycia narzędzia]\n"
            "Użytkownik: Jaka jest pogoda w Gdańsku?\n"
            "Asystent: <tool_call>\n"
            '{"name": "pobierz_pogode", "args": {"miasto": "Gdańsk"}}\n'
            "</tool_call>"
        )

        if modified_messages and isinstance(modified_messages[0], SystemMessage):
            modified_messages[0] = SystemMessage(
                content=modified_messages[0].content + tool_instruction
            )
        else:
            modified_messages.insert(0, SystemMessage(content=tool_instruction))

        return modified_messages

    def _make_api_request(self, payload: Dict[str, Any]) -> str:
        try:
            response = requests.put(
                self._api_url,
                json=payload,
                headers={'Accept': 'application/json', 'Content-Type': 'application/json'},
                timeout=60 * 5,
                **self._auth_kwargs,
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            raise ValueError(f"API error: {e}")

    def _extract_tool_calls(self, output_text: str) -> Tuple[str, List[Dict[str, Any]]]:
        tool_calls = []
        json_to_parse = None

        # 1. Correct output format
        if "<tool_call>" in output_text:
            match = re.search(r"<tool_call>(.*?)</tool_call>", output_text, re.DOTALL)
            if match:
                json_to_parse = match.group(1).strip()

        # 2. Model is stupid
        elif '"name"' in output_text and ('"args"' in output_text or '"arguments"' in output_text):
            clean_text = output_text.replace("```json", "").replace("```", "").strip()

            start_indices = [i for i, char in enumerate(clean_text) if char == '{']

            for start in start_indices:
                nesting = 0
                for i in range(start, len(clean_text)):
                    if clean_text[i] == '{':
                        nesting += 1
                    elif clean_text[i] == '}':
                        nesting -= 1

                    if nesting == 0:
                        potential_json = clean_text[start:i + 1]

                        if '"name"' in potential_json and (
                                '"args"' in potential_json or '"arguments"' in potential_json):
                            json_to_parse = potential_json
                        break

                if json_to_parse:
                    break

        if json_to_parse:
            try:
                call_data = json.loads(json_to_parse)
                arguments = call_data.get("args", call_data.get("arguments", {}))

                if isinstance(arguments, str):
                    arguments = json.loads(arguments)

                tool_calls.append({
                    "name": call_data.get("name", "unknown"),
                    "args": arguments,
                    "id": f"call_{abs(hash(output_text))}"
                })

                output_text = ""

            except json.JSONDecodeError:
                pass

        return output_text, tool_calls

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:

        tools = kwargs.get("tools")

        modified_messages = self._inject_tool_instructions(messages, tools)

        payload = {
            "messages": self._format_messages(modified_messages),
            "max_length": self._max_length,
            "temperature": self.temperature
        }

        output_text = self._make_api_request(payload)

        tool_calls = []
        if tools:
            output_text, tool_calls = self._extract_tool_calls(output_text)

        message = AIMessage(content=output_text, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


if __name__ == "__main__":
    @tool
    def add(a: int, b: int) -> int:
        "Dodaje dwie liczby ze soba"
        return a + b

    llm = BielikChatModel(temperature=0.0)
    llm_with_tools = llm.bind_tools([add])

    response = llm_with_tools.invoke([HumanMessage(content="Ile to 143412 + 4934712?")])
    print(f"Text: '{response.content}'")
    print(f"Used tools: {response.tool_calls}")