import logging

from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


logger = logging.getLogger(__name__)


class IndexesConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "delphic.indexes"

    verbose_name = _("Indexes")

    def ready(self):
        try:
            import delphic.indexes.signals  # noqa: F401
            from llama_index import set_global_service_context
            from llama_index import LLMPredictor, ServiceContext
            from langchain import OpenAI
            from django.conf import settings

            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=settings.MODEL_NAME, max_tokens=settings.MAX_TOKENS))
            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, chunk_size=512)
            logger.info(f"设置 global_service_context")
            set_global_service_context(service_context)
        except ImportError:
            pass
