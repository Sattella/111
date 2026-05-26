"""
AstrBot 长期记忆插件 (Long-Term Memory)

核心功能：
- save_long_memory FunctionTool：LLM 可主动调用保存重要记忆
- on_llm_request hook：每次 LLM 请求前自动检索相关记忆，向量相似度匹配
- 记忆按 user_id + group_id 隔离，支持跨群全局模式
- SQLite 持久化存储，支持过期时间（一周/一月/一年/永久）
- 可接入任意 OpenAI 兼容 embedding API
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .db import MemoryDB
from .embedder import Embedder


_EXPIRY_SECONDS: dict[str, float | None] = {
    "1_week":    7   * 24 * 3600,
    "1_month":   30  * 24 * 3600,
    "1_year":    365 * 24 * 3600,
    "permanent": None,
}

_EXPIRY_LABELS: dict[str, str] = {
    "1_week":    "一周",
    "1_month":   "一个月",
    "1_year":    "一年",
    "permanent": "永久",
}

# Marker injected into system_prompt to prevent double-injection in one turn
_INJECT_MARKER = "<!-- long_memory_injected -->"


@pydantic_dataclass
class SaveMemoryTool(FunctionTool[AstrAgentContext]):
    """LLM-callable tool for persisting a long-term memory entry."""

    name: str = "save_long_memory"
    description: str = (
        "将重要信息保存到用户的长期记忆中。"
        "当用户主动分享了值得记住的个人信息、偏好、习惯或事实时调用此工具，"
        "例如「用户喜欢喝绿茶」「用户的猫叫豆豆」「用户正在学 Python」。"
        "记忆按群隔离保存，后续对话会自动检索相关记忆附加到上下文。"
    )
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": (
                        "要保存的记忆，应为清晰的第三人称事实陈述，"
                        "例如：「用户喜欢喝绿茶」「用户的猫叫豆豆」"
                    ),
                },
                "expires_in": {
                    "type": "string",
                    "description": (
                        "记忆有效期。"
                        "1_week=一周（临时事件、短期计划）；"
                        "1_month=一个月（近期兴趣、短期目标）；"
                        "1_year=一年（季节性信息、年度事件）；"
                        "permanent=永久（稳定偏好、基本个人信息）"
                    ),
                    "enum": ["1_week", "1_month", "1_year", "permanent"],
                },
            },
            "required": ["content", "expires_in"],
        }
    )

    plugin: object | None = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        plugin = self.plugin
        if not plugin:
            return "❌ 插件未初始化"

        content = (kwargs.get("content") or "").strip()
        expires_in = kwargs.get("expires_in", "permanent")

        if not content:
            return "❌ 记忆内容不能为空"
        if expires_in not in _EXPIRY_SECONDS:
            expires_in = "permanent"

        event = None
        if hasattr(context, "context") and isinstance(context.context, AstrAgentContext):
            event = context.context.event
        if not event:
            return "❌ 无法获取消息上下文"

        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = str(event.message_obj.group_id or "")

        expiry_seconds = _EXPIRY_SECONDS[expires_in]
        expires_at: float | None = (
            time.time() + expiry_seconds if expiry_seconds is not None else None
        )

        embedding = await plugin.get_embedding(content)  # type: ignore[attr-defined]
        if embedding is None:
            return "❌ 无法计算向量表示，请检查 Embedding API 配置后重试"

        try:
            mid = plugin.db.add_memory(  # type: ignore[attr-defined]
                user_id=user_id,
                group_id=group_id,
                content=content,
                embedding_model=plugin.embedding_model,  # type: ignore[attr-defined]
                embedding=embedding,
                expires_at=expires_at,
            )
            label = _EXPIRY_LABELS[expires_in]
            return f"✅ 已保存记忆 #{mid}（有效期：{label}）：{content}"
        except Exception as e:
            logger.error(f"[LongMemory] 保存记忆失败: {e}")
            return f"❌ 保存失败：{e}"


class LongMemoryPlugin(Star):
    def __init__(self, context: Context, config: Any = None) -> None:
        super().__init__(context)
        self.config = config
        self._plugin_dir = Path(__file__).parent
        self.db = MemoryDB(str(self._plugin_dir / "memories.db"))
        self._embedder: Embedder | None = None
        self._load_config()
        self.db.cleanup_expired()

        if self._enable:
            self.context.add_llm_tools(SaveMemoryTool(plugin=self))
            status = "已配置" if self._embedder else "未配置 Embedding（仅支持写入）"
            logger.info(f"[LongMemory] 插件已加载，Embedding: {status}")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _cfg(self, key: str, default: Any = None) -> Any:
        if not self.config:
            return default
        return self.config.get(key, default)

    def _load_config(self) -> None:
        self._enable: bool = bool(self._cfg("enable", True))
        base_url: str = str(self._cfg("embedding_base_url", "") or "")
        self.embedding_api_key: str = str(self._cfg("embedding_api_key", "") or "")
        self.embedding_model: str = str(
            self._cfg("embedding_model", "text-embedding-3-small") or "text-embedding-3-small"
        )
        timeout: int = int(self._cfg("embedding_timeout", 30) or 30)
        self.top_k: int = max(1, int(self._cfg("top_k", 5) or 5))
        # Stored as 0-100 int in config, converted to 0.0-1.0 float
        self.similarity_threshold: float = float(
            int(self._cfg("similarity_threshold", 50) or 50)
        ) / 100.0
        self.memory_scope: str = str(
            self._cfg("memory_scope", "group_isolated") or "group_isolated"
        )
        self.inject_tool_hint: bool = bool(self._cfg("inject_tool_hint", True))

        if base_url:
            self._embedder = Embedder(
                api_url=base_url,
                api_key=self.embedding_api_key,
                model=self.embedding_model,
                timeout=timeout,
            )

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> list[float] | None:
        if not self._embedder:
            return None
        try:
            return await self._embedder.embed(text)
        except Exception as e:
            logger.error(f"[LongMemory] Embedding 请求失败: {e}")
            return None

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    async def _retrieve(
        self, user_id: str, group_id: str, query: str
    ) -> list[tuple[int, str, float]]:
        """Return list of (id, content, similarity) sorted by similarity desc."""
        query_emb = await self.get_embedding(query)
        if query_emb is None:
            return []

        scope_group = "" if self.memory_scope == "user_global" else group_id
        rows = self.db.get_active_memories(user_id, scope_group)

        results: list[tuple[int, str, float]] = []
        for row_id, content, emb_json, emb_model in rows:
            if emb_model != self.embedding_model:
                continue
            try:
                emb: list[float] = json.loads(emb_json)
                sim = self._cosine_similarity(query_emb, emb)
                if sim >= self.similarity_threshold:
                    results.append((row_id, content, sim))
            except Exception:
                continue

        results.sort(key=lambda x: -x[2])
        return results[: self.top_k]

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    @filter.on_llm_request(priority=5)
    async def on_llm_request(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> None:
        if not self._enable:
            return

        # Prevent double-injection within the same request
        sys_prompt: str = getattr(req, "system_prompt", "") or ""
        if _INJECT_MARKER in sys_prompt:
            return

        query = (event.message_str or "").strip()
        if not query:
            return

        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = str(event.message_obj.group_id or "")

        if self._embedder:
            memories = await self._retrieve(user_id, group_id, query)
        else:
            memories = []

        hint = (
            "如果用户在对话中提到了值得记住的个人信息、偏好或事实，"
            "请调用 save_long_memory 工具保存。"
            if self.inject_tool_hint
            else ""
        )

        if not memories:
            if hint:
                self._inject(req, hint)
            return

        lines = [
            "<long_term_memory>",
            "以下是关于此用户的相关历史记忆（按相关性排序）：",
        ]
        for mem_id, content, _ in memories:
            lines.append(f"  - [#{mem_id}] {content}")
        if hint:
            lines.append(hint)
        lines.append("</long_term_memory>")

        self._inject(req, "\n".join(lines))

    @staticmethod
    def _inject(req: ProviderRequest, text: str) -> None:
        """Inject text into the LLM request, preferring extra_user_content_parts."""
        try:
            parts = getattr(req, "extra_user_content_parts", None)
            if parts is not None and isinstance(parts, list):
                from astrbot.core.agent.message import TextPart

                parts.append(TextPart(text=text))
                return
        except Exception:
            pass
        try:
            req.system_prompt = (
                (req.system_prompt or "") + f"\n\n{_INJECT_MARKER}\n{text}"
            )
        except Exception as e:
            logger.error(f"[LongMemory] 注入失败: {e}")

    # ------------------------------------------------------------------
    # User commands
    # ------------------------------------------------------------------

    @filter.command("查看记忆")
    async def cmd_list(self, event: AstrMessageEvent):
        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = str(event.message_obj.group_id or "")
        scope_group = "" if self.memory_scope == "user_global" else group_id

        rows = self.db.list_memories(user_id, scope_group)

        if not rows:
            yield event.plain_result("📭 暂无长期记忆")
            return

        lines = [f"📚 长期记忆（共 {len(rows)} 条）："]
        for row_id, content, created_at, expires_at in rows:
            created = datetime.fromtimestamp(created_at).strftime("%m-%d %H:%M")
            if expires_at:
                expires = datetime.fromtimestamp(expires_at).strftime("%Y-%m-%d")
                tag = f"，{expires}到期"
            else:
                tag = "，永久"
            display = content if len(content) <= 40 else content[:37] + "..."
            lines.append(f"  #{row_id} [{created}{tag}] {display}")

        yield event.plain_result("\n".join(lines))

    @filter.command("删除记忆")
    async def cmd_delete(self, event: AstrMessageEvent, memory_id: str = ""):
        if not memory_id:
            yield event.plain_result("❌ 用法：删除记忆 <ID>")
            return
        try:
            mid = int(memory_id.strip())
        except ValueError:
            yield event.plain_result("❌ 记忆 ID 必须是数字")
            return

        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = str(event.message_obj.group_id or "")
        scope_group = "" if self.memory_scope == "user_global" else group_id

        ok = self.db.delete_memory(mid, user_id, scope_group)
        if ok:
            yield event.plain_result(f"✅ 已删除记忆 #{mid}")
        else:
            yield event.plain_result(f"❌ 未找到记忆 #{mid}（只能删除自己的记忆）")

    @filter.command("清空记忆")
    async def cmd_clear(self, event: AstrMessageEvent):
        user_id = str(event.get_sender_id() or event.unified_msg_origin)
        group_id = str(event.message_obj.group_id or "")
        scope_group = "" if self.memory_scope == "user_global" else group_id

        count = self.db.clear_memories(user_id, scope_group)
        yield event.plain_result(f"✅ 已清空 {count} 条记忆")

    # ------------------------------------------------------------------

    async def terminate(self) -> None:
        self.db.cleanup_expired()
        logger.info("[LongMemory] 插件已卸载")
