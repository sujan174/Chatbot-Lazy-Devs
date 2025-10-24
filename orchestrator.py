import os
import json
import asyncio
import traceback
from typing import Any, Dict, List, Optional, Tuple
import google.generativeai as genai
import google.generativeai.protos as protos
from dotenv import load_dotenv
from pathlib import Path
import importlib.util
from datetime import datetime

load_dotenv()

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

genai.configure(api_key=GOOGLE_API_KEY)


class OrchestratorAgent:
    """
    Intelligent orchestration agent that coordinates specialized sub-agents.
    Optimized for speed, natural conversation, and FastAPI integration.
    """
    
    def __init__(
        self, 
        user_id: str, 
        connectors_dir: str = "connectors", 
        verbose: bool = False,
        max_iterations: int = 10
    ):
        self.user_id = user_id
        self.connectors_dir = Path(connectors_dir)
        self.sub_agents: Dict[str, Any] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.verbose = verbose
        self.max_iterations = max_iterations
        
        # Pre-initialized state
        self._initialized = False
        self.model = None
        self.chat = None
        self.conversation_history = []
        
        # Performance tracking
        self._agent_load_time = None
        
        self.system_prompt = """You are a helpful AI assistant with access to specialized tools and agents.

You can help users with:
- Jira: Create, update, search tickets and manage issues
- Slack: Send messages, read channels, manage conversations
- Notion: Create, update, and search pages and databases

Guidelines:
- Respond naturally and conversationally - you're chatting with a user, not executing a script
- Only use tools when the user asks you to DO something (create, update, send, etc.)
- For general questions or casual conversation, just respond directly without using tools
- When using tools, be efficient - don't make unnecessary calls
- If a task requires multiple steps, chain them logically
- Always confirm what you've done and provide relevant details
- If something goes wrong, explain clearly and offer alternatives

Remember: You're here to help, not just to route requests. Be friendly, concise, and helpful."""

    async def initialize(self):
        """
        Pre-load all agents and prepare the model.
        Call this once at startup for better performance.
        """
        if self._initialized:
            return
        
        start_time = datetime.now()
        
        if self.verbose:
            print(f"[{self.user_id}] Initializing orchestrator...")
        
        await self._discover_and_load_agents()
        
        if self.sub_agents:
            self._initialize_model()
        
        self._agent_load_time = (datetime.now() - start_time).total_seconds()
        self._initialized = True
        
        if self.verbose:
            print(f"[{self.user_id}] Initialization complete in {self._agent_load_time:.2f}s")

    async def _discover_and_load_agents(self):
        """Load all agent connectors from the connectors directory."""
        if not self.connectors_dir.exists():
            if self.verbose:
                print(f"[{self.user_id}] Creating connectors directory...")
            self.connectors_dir.mkdir(parents=True, exist_ok=True)
            return
        
        connector_files = list(self.connectors_dir.glob("*_agent.py"))
        
        if not connector_files:
            if self.verbose:
                print(f"[{self.user_id}] No agent connectors found")
            return
        
        # Load agents in parallel for faster startup
        load_tasks = [
            self._load_single_agent(connector_file) 
            for connector_file in connector_files
        ]
        
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if r is True)
        if self.verbose:
            print(f"[{self.user_id}] Loaded {successful}/{len(connector_files)} agents")

    async def _load_single_agent(self, connector_file: Path) -> bool:
        """Load a single agent connector."""
        agent_name = connector_file.stem.replace("_agent", "")
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"connectors.{agent_name}_agent",
                connector_file
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if not hasattr(module, 'Agent'):
                if self.verbose:
                    print(f"[{self.user_id}] No 'Agent' class in {connector_file.name}")
                return False
            
            agent_class = module.Agent
            # MODIFIED: Pass the orchestrator's user_id to the agent's constructor
            agent_instance = agent_class(user_id=self.user_id)
            await agent_instance.initialize()
            
            capabilities = await agent_instance.get_capabilities()
            
            self.sub_agents[agent_name] = agent_instance
            self.agent_capabilities[agent_name] = capabilities
            
            if self.verbose:
                print(f"[{self.user_id}] ✓ Loaded {agent_name} ({len(capabilities)} capabilities)")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"[{self.user_id}] ✗ Failed to load {agent_name}: {e}")
                traceback.print_exc()
            return False

    def _initialize_model(self):
        """Initialize the Gemini model with agent tools."""
        agent_tools = self._create_agent_tools()
        
        self.model = genai.GenerativeModel(
            'models/gemini-2.0-flash-exp',  # Faster model
            system_instruction=self.system_prompt,
            tools=agent_tools
        )
        
        self.chat = self.model.start_chat(history=self.conversation_history)
    
    def _create_agent_tools(self) -> List[protos.FunctionDeclaration]:
        """Create function declarations for each agent."""
        tools = []
        
        for agent_name, capabilities in self.agent_capabilities.items():
            # Create a concise, natural description
            capabilities_str = "\n".join(f"• {cap}" for cap in capabilities)
            
            tool = protos.FunctionDeclaration(
                name=f"use_{agent_name}",
                description=f"""Access the {agent_name.title()} agent.

Capabilities:
{capabilities_str}

Use this when the user wants to interact with {agent_name.title()}.""",
                parameters=protos.Schema(
                    type_=protos.Type.OBJECT,
                    properties={
                        "instruction": protos.Schema(
                            type_=protos.Type.STRING,
                            description=f"What to do in {agent_name.title()} (be specific and clear)"
                        ),
                        "context": protos.Schema(
                            type_=protos.Type.STRING,
                            description="Additional context or data from previous steps (optional)"
                        )
                    },
                    required=["instruction"]
                )
            )
            tools.append(tool)
        
        return tools
    
    async def _call_sub_agent(
        self, 
        agent_name: str, 
        instruction: str, 
        context: Optional[str] = None
    ) -> str:
        """Execute a sub-agent with the given instruction."""
        if agent_name not in self.sub_agents:
            return f"Error: {agent_name} agent is not available"
        
        try:
            agent = self.sub_agents[agent_name]
            
            # Combine instruction with context if provided
            full_instruction = instruction
            if context:
                full_instruction = f"Context: {context}\n\nTask: {instruction}"
            
            if self.verbose:
                print(f"[{self.user_id}] → Calling {agent_name}: {instruction[:100]}...")
            
            result = await agent.execute(full_instruction)
            
            if self.verbose:
                print(f"[{self.user_id}] ← {agent_name} completed")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in {agent_name}: {str(e)}"
            if self.verbose:
                print(f"[{self.user_id}] ✗ {error_msg}")
                traceback.print_exc()
            return error_msg
    
    async def chat_async(self, message: str) -> str:
        """
        Send a message and get a response.
        Main entry point for conversations.
        
        Args:
            message: User's message
            
        Returns:
            Assistant's response
        """
        # Lazy initialization if not already done
        if not self._initialized:
            await self.initialize()
        
        if not self.sub_agents:
            return "I'm sorry, but I don't have any specialized tools available right now. Please make sure agent connectors are properly configured."
        
        try:
            return await self._process_message(message)
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}"
            if self.verbose:
                print(f"[{self.user_id}] Error: {e}")
                traceback.print_exc()
            return error_msg
    
    async def _process_message(self, user_message: str) -> str:
        """Process a message through the orchestration loop."""
        # Send initial message
        response = await self.chat.send_message_async(user_message)
        
        # Handle function calling loop
        iteration = 0
        while iteration < self.max_iterations:
            parts = response.candidates[0].content.parts
            
            # Check if there are any function calls
            function_calls = [
                part.function_call 
                for part in parts 
                if hasattr(part, 'function_call') and part.function_call
            ]
            
            if not function_calls:
                # No more function calls, we're done
                break
            
            # Process all function calls in parallel for speed
            function_responses = await self._execute_function_calls(function_calls)
            
            # Send all responses back to the model
            response = await self.chat.send_message_async(
                genai.protos.Content(parts=function_responses)
            )
            
            iteration += 1
        
        if iteration >= self.max_iterations:
            if self.verbose:
                print(f"[{self.user_id}] ⚠ Reached max iterations")
        
        # Update conversation history
        self.conversation_history = self.chat.history
        
        return response.text
    
    async def _execute_function_calls(
        self, 
        function_calls: List[Any]
    ) -> List[genai.protos.Part]:
        """Execute multiple function calls in parallel."""
        tasks = []
        
        for fc in function_calls:
            # Extract agent name from tool name
            tool_name = fc.name
            if tool_name.startswith("use_"):
                agent_name = tool_name[4:]  # Remove "use_" prefix
            else:
                agent_name = tool_name
            
            # Convert proto args to dict
            args = self._proto_to_dict(fc.args)
            instruction = args.get("instruction", "")
            context = args.get("context")
            
            # Create task
            task = self._call_sub_agent(agent_name, instruction, context)
            tasks.append((tool_name, task))
        
        # Execute all in parallel
        results = await asyncio.gather(*[task for _, task in tasks])
        
        # Create function response parts
        response_parts = []
        for (tool_name, _), result in zip(tasks, results):
            response_parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tool_name,
                        response={"result": result}
                    )
                )
            )
        
        return response_parts
    
    def _proto_to_dict(self, value: Any) -> Any:
        """Convert protobuf types to Python native types."""
        type_str = str(type(value))
        
        if "MapComposite" in type_str:
            return {k: self._proto_to_dict(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._proto_to_dict(item) for item in value]
        else:
            return value
    
    def get_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history in a simple format.
        Useful for debugging or displaying chat history.
        
        Returns:
            List of messages with 'role' and 'content' keys
        """
        history = []
        for msg in self.conversation_history:
            role = msg.role
            content = ""
            
            for part in msg.parts:
                if hasattr(part, 'text') and part.text:
                    content += part.text
            
            if content:
                history.append({"role": role, "content": content})
        
        return history
    
    def clear_history(self):
        """Clear conversation history and start fresh."""
        self.conversation_history = []
        if self.chat:
            self.chat = self.model.start_chat(history=[])
        if self.verbose:
            print(f"[{self.user_id}] Conversation history cleared")
    
    async def cleanup(self):
        """Cleanup all resources."""
        if self.verbose:
            print(f"[{self.user_id}] Cleaning up...")
        
        cleanup_tasks = []
        for agent_name, agent in self.sub_agents.items():
            if hasattr(agent, 'cleanup'):
                cleanup_tasks.append(agent.cleanup())
        
        if cleanup_tasks:
            results = await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            
            for agent_name, result in zip(self.sub_agents.keys(), results):
                if isinstance(result, Exception):
                    if self.verbose:
                        print(f"[{self.user_id}] ✗ Error cleaning up {agent_name}: {result}")
        
        self._initialized = False
        if self.verbose:
            print(f"[{self.user_id}] Cleanup complete")
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        if self._initialized and self.sub_agents:
            # Note: async cleanup in __del__ is tricky, best to call cleanup() explicitly
            pass


# Convenience function for simple usage
async def create_orchestrator(
    user_id: str, 
    connectors_dir: str = "connectors",
    verbose: bool = False
) -> OrchestratorAgent:
    """
    Create and initialize an orchestrator agent.
    
    Usage:
        orchestrator = await create_orchestrator("user_123")
        response = await orchestrator.chat_async("Create a Jira ticket")
    """
    agent = OrchestratorAgent(user_id, connectors_dir, verbose)
    await agent.initialize()
    return agent


