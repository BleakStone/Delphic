import json

from channels.generic.websocket import AsyncWebsocketConsumer

from delphic.utils.collections import load_collection_model
from delphic.utils.paths import extract_connection_id, extract_graph_id
            
from llama_index import get_response_synthesizer
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore
from llama_index.query_engine import KnowledgeGraphQueryEngine


class CollectionNebulaQueryConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("Connecting...")  # Debugging print statement
        try:

            self.graph_id = extract_graph_id(self.scope["path"])
            print(f"连接图id: {self.graph_id}")
            # define LLM
            # todo 图的结构需要从mysql表中，或者前端传递过来
            space_name = self.graph_id
            graph_store = NebulaGraphStore(space_name=space_name)
            storage_context = StorageContext.from_defaults(graph_store=graph_store)

            query_engine = KnowledgeGraphQueryEngine(
                storage_context=storage_context,
                # service_context=service_context,
                # llm=llm,
                verbose=True,
            )
            
            # NOTE: lazy import
            from llama_index.chat_engine import CondenseQuestionChatEngine
            self.custom_chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=query_engine,
            )

            print(f"Chat_engine loaded: {self.custom_chat_engine}")
            await self.accept()
            print("Connected.")  # Debugging print statement
        except ValueError as e:
            print(f"Value error prevented model loading: {e}")
            await self.accept()
            await self.close(code=4000)
        except Exception as e:
            print(f"Error during connection: {e}")  # Debugging print statement

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        print(f"Received text_data_json: {text_data_json}")

        if self.custom_chat_engine is not None:
            query_str = text_data_json["query"]
            modified_query_str = f"""
            Please return a nicely formatted markdown string to this request:

            {query_str}
            """
            
            # response = self.index.query(modified_query_str)
            # query_engine = self.index.as_query_engine()
            response = self.custom_chat_engine.chat(query_str)

            # Format the response as markdown
            markdown_response = f"## Response\n\n{response}\n\n"
            if response.sources:
                markdown_sources = f"## Sources\n\n{response.sources}"
            else:
                markdown_sources = ""

            formatted_response = f"{markdown_response}{markdown_sources}"

            await self.send(json.dumps({"response": formatted_response}, indent=4))
        else:
            await self.send(
                json.dumps({"error": "No index loaded for this connection."}, indent=4)
            )
