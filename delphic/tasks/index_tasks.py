import logging
import os
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files import File
from langchain import OpenAI
from llama_index import (
    SimpleDirectoryReader,
    LLMPredictor,
    ServiceContext,
    download_loader,
    VectorStoreIndex,
    KnowledgeGraphIndex,
)
from llama_index.storage.storage_context import StorageContext
from llama_index.graph_stores import NebulaGraphStore

from config import celery_app
from delphic.indexes.models import Collection, CollectionStatus


logger = logging.getLogger(__name__)

# to do 需要更具文件大小设置，特别是构建graphindex极其耗时
@celery_app.task(soft_time_limit=3000)
def create_index(collection_id):
    """
    Celery task to create a GPTSimpleVectorIndex for a given Collection object.

    This task takes the ID of a Collection object, retrieves it from the
    database along with its related documents, and saves the document files
    to a temporary directory. Then, it creates a GPTSimpleVectorIndex using
    the provided code and saves the index to the Comparison.model FileField.

    Args:
        collection_id (int): The ID of the Collection object for which the
                             index should be created.

    Returns:
        bool: True if the index is created and saved successfully, False otherwise.
    """
    try:
        # Get the Collection object with related documents
        collection = Collection.objects.prefetch_related("documents").get(
            id=collection_id
        )
        collection.status = CollectionStatus.RUNNING
        collection.save()

        try:
            # Create a temporary directory to store the document files
            with tempfile.TemporaryDirectory() as tempdir:
                tempdir_path = Path(tempdir)

                # Save the document files to the temporary directory
                for document in collection.documents.all():
                    logger.info(f"filename:{document.file.name}")
                    logger.info(f"filepath:{document.file.path}")
                    with document.file.open("rb") as f:
                        file_data = f.read()

                    temp_file_path = tempdir_path / document.file.name
                    temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                    with temp_file_path.open("wb") as f:
                        f.write(file_data)

                # todo chunk_size_limit过期 使用配置或传参，以及全局service_context的用法
                # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=settings.MODEL_NAME, max_tokens=settings.MAX_TOKENS))
                # service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size_limit=512)

                logger.info(f"path:{str(tempdir_path)}")
                documents = SimpleDirectoryReader(input_dir=str(tempdir_path), recursive=True).load_data()
                # documents = SimpleDirectoryReader(input_files==[str(temp_file_path)]).load_data()
                logger.info(f"len:{len(documents)}")
                # 后期可以将index构建日志返回给前端。在构建greahindex较费时
                # index = VectorStoreIndex.from_documents(documents, service_context=service_context)
                index = VectorStoreIndex.from_documents(documents)
                #./storage 存入本地时生产环境需要注意docker销毁的问题，最好也同时存入数据库
                index.storage_context.persist(persist_dir=f"./storage/{collection_id}/vectorindex")
                logger.info(f"构建VectorStoreIndex完成")

                # 构建graphindexß
                space_name = "llamaindex"
                edge_types, rel_prop_names = ["relationship"], ["relationship"] # default, could be omit if create from an empty kg
                tags = ["entity"] # default, could be omit if create from an empty kg
                graph_store = NebulaGraphStore(space_name=space_name, edge_types=edge_types, rel_prop_names=rel_prop_names, tags=tags)
                kg_storage_context = StorageContext.from_defaults(graph_store=graph_store)
                kg_index = KnowledgeGraphIndex.from_documents(
                    documents,
                    storage_context=kg_storage_context,
                    max_triplets_per_chunk=10,
                    # service_context=service_context,
                    space_name=space_name,
                    edge_types=edge_types,
                    rel_prop_names=rel_prop_names,
                    tags=tags,
                    include_embeddings=True,
                )
                kg_index.storage_context.persist(persist_dir=f"./storage/{collection_id}/graphindex")
                logger.info(f"构建KnowledgeGraphIndex完成")

                collection.status = CollectionStatus.COMPLETE
                collection.model.name = "sucess"
                collection.save()

                # 将index文件存入数据库
                # index_str = index.save_to_string()
                # # Save the index_str to the Comparison.model FileField
                # with tempfile.NamedTemporaryFile(delete=False) as f:
                #     f.write(index_str.encode())
                #     f.flush()
                #     f.seek(0)
                #     collection.model.save(f"model_{uuid.uuid4()}.json", File(f))
                #     collection.status = CollectionStatus.COMPLETE
                #     collection.save()

                # Delete the temporary index file
                os.unlink(f.name)

            collection.processing = False
            collection.save()

            return True

        except Exception as e:
            logger.error(f"Error creating index for collection {collection_id}: {e}")
            logger.exception(f"exception creating index for collection {collection_id}: {e}")

            collection.status = CollectionStatus.ERROR
            collection.save()

            return False

    except Exception as e:
        logger.error(f"Error loading collection: {e}")
        return False
