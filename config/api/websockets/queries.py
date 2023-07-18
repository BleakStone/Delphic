import json

from channels.generic.websocket import AsyncWebsocketConsumer

from delphic.utils.collections import load_collection_model
from delphic.utils.paths import extract_connection_id
            
from llama_index import get_response_synthesizer
from llama_index.query_engine import RetrieverQueryEngine
# Retrievers 
from llama_index.retrievers import BaseRetriever, VectorIndexRetriever, KGTableRetriever

from delphic.utils.retriever import CustomRetriever


class CollectionQueryConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        print("Connecting...")  # Debugging print statement
        try:
            self.collection_id = extract_connection_id(self.scope["path"])
            print(f"Connect to collection model: {self.collection_id}")
            # define LLM
            self.vector_index, self.kg_index = await load_collection_model(self.collection_id)
            print(f"Index loaded: {self.vector_index}, {self.kg_index}")
            # self.query_engine = self.index.as_query_engine()
            # todo 需要看react和 condense_question两种模式的效果,以及对上文历史问题的契合度
            self.vector_chat_engine = self.vector_index.as_chat_engine(chat_mode="condense_question", verbose=True)
            self.kg_chat_engine = self.kg_index.as_chat_engine(
                chat_mode="condense_question",
                verbose=True,
                include_text=False,  
                retriever_mode='keyword',
                response_mode="tree_summarize",
                )

            # create custom retriever构建自定义查询引擎
            vector_retriever = VectorIndexRetriever(index=self.vector_index)
            kg_retriever = KGTableRetriever(index=self.kg_index, retriever_mode='keyword', include_text=False)
            custom_retriever = CustomRetriever(vector_retriever, kg_retriever)
            # create response synthesizer
            response_synthesizer = get_response_synthesizer(
                # service_context=service_context,
                response_mode="tree_summarize",
            )
            custom_query_engine = RetrieverQueryEngine(
                retriever=custom_retriever,
                response_synthesizer=response_synthesizer,
            )
            # NOTE: lazy import
            from llama_index.chat_engine import CondenseQuestionChatEngine
            self.custom_chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=custom_query_engine,
                # service_context=service_context,
            )

            print(f"Chat_engine loaded: {self.vector_chat_engine}, {self.kg_chat_engine}")
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

        if self.kg_index is not None:
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
