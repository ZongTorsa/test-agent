# AGENTS.md

## 项目定位
Python Agent & RAG 学习工程，边做边学智能体开发。

## 技术栈
langchain、langchain-openai、chromadb、python-dotenv

## 编码规范
1. 代码模块化拆分，禁止全部写在一个文件里
2. 自动维护 requirements.txt，新增依赖就更新
3. 模型调用走 Codex 内置能力，不需要手动配置密钥
4. 每次改动后给出运行命令和自测示例
5. 代码加注释，新增功能先说明改了哪些文件

## 行为约束
1. 不删除已有代码，只做增量迭代
2. 代码报错优先自己修复，修不了再说明卡在哪
3. 任务完成后简要总结实现思路