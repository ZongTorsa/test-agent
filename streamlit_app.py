"""Streamlit 测试面板：用于交互式验证多智能体、RAG 与记忆能力。"""

from __future__ import annotations

import os

import streamlit as st
from openai import OpenAIError

from config import get_settings
from memory import MultiAgentMemory, retrieve_long_term_memory
from multi_agent import run_multi_agent_workflow
from vector_store import build_vector_store, search_similar_documents


st.set_page_config(page_title="Agent & RAG 测试台", page_icon="🧠", layout="wide")


def get_session_memory() -> MultiAgentMemory:
    """为每个浏览器会话保留独立的短期记忆。"""
    if "workflow_memory" not in st.session_state:
        st.session_state.workflow_memory = MultiAgentMemory()
    return st.session_state.workflow_memory


def show_environment_status() -> None:
    """展示凭据与本地目录状态，不展示任何密钥值。"""
    settings = get_settings()
    left, middle, right = st.columns(3)
    left.metric("DeepSeek 凭据", "已检测到" if settings.deepseek_api_key else "缺失")
    middle.metric("DashScope 凭据", "已检测到" if settings.dashscope_api_key else "缺失")
    right.metric("向量库目录", "已就绪" if settings.chroma_dir.exists() else "未构建")


def render_multi_agent_tab() -> None:
    """渲染 Planner → Executor → 汇总 的交互测试页。"""
    st.subheader("多智能体协同测试")
    request = st.text_area(
        "复杂需求",
        value="整理知识库中所有关于 RAG 优化的内容并总结优缺点",
        height=120,
    )
    if st.button("运行 Planner + Executor", type="primary", use_container_width=True):
        if not request.strip():
            st.warning("请输入需要执行的需求。")
            return
        try:
            with st.spinner("Planner 正在拆解任务，Executor 正在依次执行..."):
                plan, results, summary = run_multi_agent_workflow(
                    request, get_session_memory()
                )
            st.success("工作流执行完成，结果已写入短期和长期记忆。")
            with st.expander("Planner 执行计划", expanded=True):
                for task in plan.tasks:
                    st.write(f"{task.task_id}. {task.instruction}")
            with st.expander("Executor 任务结果", expanded=True):
                for item in results:
                    st.markdown(f"#### 任务 {item.task_id}")
                    st.caption(item.instruction)
                    st.write(item.result)
            st.markdown("### 最终汇总")
            st.write(summary)
        except (OpenAIError, RuntimeError, ValueError) as error:
            st.error(f"工作流执行失败：{error}")


def render_rag_tab() -> None:
    """渲染知识库重建和语义检索测试页。"""
    st.subheader("RAG 向量库测试")
    left, right = st.columns([2, 1])
    with right:
        if st.button("重建 knowledge 向量库", use_container_width=True):
            try:
                with st.spinner("正在加载、切分并写入 Chroma..."):
                    store = build_vector_store()
                st.success(f"向量库重建完成：{store._collection.count()} 个片段。")
            except (OpenAIError, RuntimeError, ValueError, FileNotFoundError) as error:
                st.error(f"重建失败：{error}")
    with left:
        query = st.text_input("检索问题", value="RAG 的基本流程是什么？")
        count = st.slider("返回片段数", min_value=1, max_value=5, value=3)
        if st.button("检索知识库", use_container_width=True):
            try:
                documents = search_similar_documents(query, k=count)
                if not documents:
                    st.info("知识库未找到相关资料。")
                for index, document in enumerate(documents, start=1):
                    st.markdown(f"#### 片段 {index} · {document.metadata.get('source', '未知来源')}")
                    st.code(document.page_content, language="markdown")
            except (OpenAIError, RuntimeError, ValueError, FileNotFoundError) as error:
                st.error(f"检索失败：{error}")


def render_memory_tab() -> None:
    """渲染当前会话短期记忆和跨会话长期记忆查看页。"""
    st.subheader("双层记忆测试")
    memory = get_session_memory()
    left, right = st.columns(2)
    with left:
        st.markdown("#### 短期记忆（当前网页会话）")
        st.text(memory.short_term.format_context())
        if st.button("清空当前会话短期记忆", use_container_width=True):
            st.session_state.workflow_memory = MultiAgentMemory()
            st.rerun()
    with right:
        st.markdown("#### 长期记忆（Chroma / project_memory）")
        query = st.text_input("历史项目事实检索", value="项目使用什么向量库和嵌入模型？")
        if st.button("检索长期记忆", use_container_width=True):
            facts = retrieve_long_term_memory(query)
            if facts:
                st.success(f"命中 {len(facts)} 条历史事实")
                for fact in facts:
                    st.write(f"- {fact}")
            else:
                st.info("尚未找到相关长期记忆。请先运行一次多智能体工作流。")


def main() -> None:
    """渲染 Streamlit 应用。"""
    st.title("🧠 Python Agent & RAG 测试台")
    st.caption("测试多智能体协作、RAG 检索与双层记忆；不会显示 API Key。")
    show_environment_status()
    st.divider()
    multi_agent_tab, rag_tab, memory_tab = st.tabs(["多智能体", "RAG 检索", "双层记忆"])
    with multi_agent_tab:
        render_multi_agent_tab()
    with rag_tab:
        render_rag_tab()
    with memory_tab:
        render_memory_tab()


if __name__ == "__main__":
    main()
