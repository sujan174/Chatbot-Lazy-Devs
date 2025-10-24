import os
import json
import asyncio
import traceback
from typing import Any, Dict, List, Optional
import google.generativeai as genai
import google.generativeai.protos as protos
from dotenv import load_dotenv
from pathlib import Path
import importlib.util

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)


class OrchestratorAgent:
    def __init__(self, user_id: str, connectors_dir: str = "connectors", verbose: bool = False):
        self.user_id = user_id
        self.connectors_dir = Path(connectors_dir)
        self.sub_agents: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.verbose = verbose
        
        self.system_prompt = """
        You are an intelligent orchestration agent that coordinates multiple specialized AI agents.
        
        Your responsibilities:
        1. Understand user requests and break them down into sub-tasks
        2. Determine which specialized agent(s) should handle each sub-task
        3. Execute tasks in the correct order, using outputs from one agent as inputs to another
        4. Synthesize results from multiple agents into coherent responses
        5. Handle errors gracefully and provide clear feedback
        
        Available specialized agents and their capabilities will be provided to you as tools.
        Each tool represents a complete agent that can handle complex requests in its domain.
        
        When working with agents:
        - Be specific in your instructions to sub-agents
        - Chain operations when one task depends on another
        - Confirm critical actions before executing them
        - Provide clear summaries of what was accomplished
        """
        
        self.model = None
        self.chat = None
        self.conversation_history = []
        
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }

    async def discover_and_load_agents(self):
        if not self.connectors_dir.exists():
            print(f"❌ Connectors directory '{self.connectors_dir}' not found!")
            print(f"Creating directory...")
            self.connectors_dir.mkdir(parents=True, exist_ok=True)
            return
        
        connector_files = list(self.connectors_dir.glob("*_agent.py"))
        
        if not connector_files:
            print(f"⚠ No agent connectors found in '{self.connectors_dir}'")
            print(f"  Expected files matching pattern: *_agent.py")
            return
        
        for connector_file in connector_files:
            agent_name = connector_file.stem.replace("_agent", "")
            
            try:
                spec = importlib.util.spec_from_file_location(
                    f"connectors.{agent_name}_agent",
                    connector_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                if hasattr(module, 'Agent'):
                    agent_class = module.Agent
                    agent_instance = agent_class() 
                    await agent_instance.initialize()
                    
                    capabilities = await agent_instance.get_capabilities()
                    
                    self.sub_agents[agent_name] = agent_instance
                    self.agent_capabilities[agent_name] = capabilities
                    
                else:
                    print(f"  ✗ No 'Agent' class found in {connector_file}")
                    
            except Exception as e:
                print(f"  ✗ Failed to load {agent_name}: {e}")
                traceback.print_exc()
        
        print(f"\n✅ Loaded {len(self.sub_agents)} agent(s) successfully for User: {self.user_id}.")

    
    def _create_agent_tools(self) -> List[protos.FunctionDeclaration]:
        tools = []
        
        for agent_name, capabilities in self.agent_capabilities.items():
            tool = protos.FunctionDeclaration(
                name=f"use_{agent_name}_agent",
                description=f"""Use the {agent_name} agent to perform tasks. 
                
Capabilities: {', '.join(capabilities)}

This agent can handle complex requests related to {agent_name}. 
Provide a clear instruction describing what you want to accomplish.""",
                parameters=protos.Schema(
                    type_=protos.Type.OBJECT,
                    properties={
                        "instruction": protos.Schema(
                            type_=protos.Type.STRING,
                            description=f"Clear instruction for the {agent_name} agent"
                        ),
                        "context": protos.Schema(
                            type_=protos.Type.OBJECT, 
                            description="Optional context or data from previous steps"
                        )
                    },
                    required=["instruction"]
                )
            )
            tools.append(tool)
        
        return tools
    
    async def call_sub_agent(self, agent_name: str, instruction: str, context: Any = None) -> str:
        if agent_name not in self.sub_agents:
            return f"Error: Agent '{agent_name}' not found"
        
        context_str = ""
        if context:
            if isinstance(context, (dict, list)): 
                context_str = json.dumps(context, indent=2)
            else:
                context_str = str(context)
        
        try:
            agent = self.sub_agents[agent_name]
            
            full_instruction = instruction
            if context_str:
                full_instruction = f"Context from previous steps:\n{context_str}\n\nTask: {instruction}"
            
            result = await agent.execute(full_instruction)
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing {agent_name} agent: {str(e)}"
            print(f"✗ {error_msg}")
            traceback.print_exc()
            return error_msg
    
    async def process_message(self, user_message: str) -> str:
        if not self.chat:
            await self.discover_and_load_agents()
            
            if not self.sub_agents:
                return "No agents available. Please add agent connectors to the 'connectors' directory."
            
            agent_tools = self._create_agent_tools()
            
            self.model = genai.GenerativeModel(
                'models/gemini-2.5-pro',
                system_instruction=self.system_prompt,
                tools=agent_tools
            )
            
            self.chat = self.model.start_chat(history=self.conversation_history)
        
        response = await self.chat.send_message_async(user_message)
        
        max_iterations = 15
        iteration = 0
        
        while iteration < max_iterations:
            parts = response.candidates[0].content.parts
            has_function_call = any(
                hasattr(part, 'function_call') and part.function_call 
                for part in parts
            )
            
            if not has_function_call:
                break
            
            function_call = None
            for part in parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    break
            
            if not function_call:
                break
            
            tool_name = function_call.name
            
            if tool_name.startswith("use_") and tool_name.endswith("_agent"):
                agent_name = tool_name[4:-6]
            else:
                agent_name = tool_name
            
            args = self._deep_convert_proto_args(function_call.args)
            
            instruction = args.get("instruction", "")
            context = args.get("context", "")
            
            result = await self.call_sub_agent(agent_name, instruction, context)
            
            response = await self.chat.send_message_async(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result}
                        )
                    )]
                )
            )
            
            iteration += 1
        
        if iteration >= max_iterations:
            print(f"⚠ Warning: Reached maximum orchestration iterations for {self.user_id}")
        
        self.conversation_history = self.chat.history
        return response.text
    
    def _deep_convert_proto_args(self, value: Any) -> Any:
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value
    
    async def cleanup(self):
        for agent_name, agent in self.sub_agents.items():
            try:
                if hasattr(agent, 'cleanup'):
                    await agent.cleanup()
            except Exception as e:
                print(f"  ✗ Error shutting down {agent_name}: {e}")

