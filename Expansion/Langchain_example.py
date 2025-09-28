
from langchain.agents import AgentType, initialize_agent,Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline

import torch
import json
import os

class Qwen_AIAgent:
    def __init__(self,model,device='cuda'):
        '''
        AI Agent based Qwen model

        Args:
            model: model name in huggingface hub
            device
        '''

        self.model_name = model
        self.device = device
        self.llm = self._setup_qwen_model()
        self.memory = self._setup_memory()
        self.tools = self._initialize_tools()
        self.agent = self._create_agent()

    def _setup_qwen_model(self):
        ''' Setup Qwen model from Huggingface hub'''

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        pipe = pipeline(
            'text-generation',
            tokenizer=tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=True,
            return_full_text=False
        )

        llm = HuggingFacePipeline(pipeline=pipe)
        return llm

    def _setup_memory(self):

        memory_ = ConversationBufferMemory(
            memory_key='chat_history',
            k = 10,     # save last 10 interactions
            return_messages=True,
        )
        return memory_

    def _initialize_tools(self):
        ''' Initialize tools for the agent'''

        tools = [
            Tool(
                name = 'File Processor',
                func = self.file_processor_tool,
                description = 'A tool to read, analyze, and summarize the content of text files. The input should be .json file with file_path and operation'
            ),
            Tool(
                name = 'Code Executor',
                func = self.code_executor_tool,
                description = 'Execute python code to do calculations or data processing. Input should be a valid python code string.'
            ),
            Tool(
                name = 'Web Search',
                func = self.web_search_tool,
                description = 'A tool to search the web for current information. Input should be a search query string.'
            ),
            Tool(
                name = 'Database search',
                func = self.database_search_tool,
                description = 'A tool to search a database for specific information. Input should be SQL query string.'
            ),
            Tool(
                name = 'API Caller',
                func = self.api_caller_tool,
                description = 'A tool to call external APIs. Input should be a valid API endpoint and parameters in JSON format.'
            )
        ]

    def _create_agent(self):

        system_prompt ='''你是一个基于Qwen的AI助手，具有强大的工具调用能力。请遵循以下规则：
            1.仔细分析用户需求，制定合理的执行计划
            2.优先使用合适的工具完成任务
            3.确保工具调用的输入格式正确
            4.对于复杂任务，分步执行并整合结果
            5.保持回答的准确性。
            
            可用工具：{tools}
            '''

        agent = initialize_agent(
            tools = self.tools,
            llm = self.llm,
            agent = AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory = self.memory,
            verbose = True,
            max_iterations = 5,
            handle_parsing_error = True,
            agent_kwargs={
                'system_message': SystemMessage(content = system_prompt)
            }
        )
        return agent

    def _file_processor_tool(self,input_str:str)->str:
        pass
        




