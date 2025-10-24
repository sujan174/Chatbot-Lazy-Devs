import os
import sys
import json
from typing import Any, Dict, List

import google.generativeai as genai
import google.generativeai.protos as protos
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add parent directory to path to import base_agent
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from base_agent import BaseAgent


class Agent(BaseAgent):
    """Specialized agent for GitHub operations via MCP"""
    
    def __init__(self, user_id: str):
        super().__init__(user_id)
        # self.name is automatically set to "github" by BaseAgent
        self.session: ClientSession = None
        self.stdio_context = None
        self.model = None
        self.available_tools = []
        
        self.schema_type_map = {
            "string": protos.Type.STRING,
            "number": protos.Type.NUMBER,
            "integer": protos.Type.INTEGER,
            "boolean": protos.Type.BOOLEAN,
            "object": protos.Type.OBJECT,
            "array": protos.Type.ARRAY,
        }
        
        self.system_prompt = """
        You are a GitHub specialist agent. You have access to tools to:
        - Read repositories, files, and branches
        - Manage issues and pull requests (create, list, comment, close)
        - Search for code and commits
        - Get user and organization details
        
        Always:
        1. Be precise with repository names (owner/repo).
        2. Confirm issue or PR numbers before acting on them.
        3. Summarize actions taken (e.g., "I've created issue #123 in 'owner/repo'").
        4. Format code snippets and file paths correctly.
        """
    
    async def initialize(self):
        """Connect to the GitHub MCP server"""
        try:
            # --- START: EDITED BLOCK ---
            # Use the local npx server package, just like the slack_agent
            # This avoids the complex auth and DNS issues of mcp-remote
            print("Attempting to launch @modelcontextprotocol/server-github via npx...")
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"], 
                env={**os.environ}
            )
            # --- END: EDITED BLOCK ---
            
            self.stdio_context = stdio_client(server_params)
            stdio, write = await self.stdio_context.__aenter__()
            self.session = ClientSession(stdio, write)
            
            await self.session.__aenter__()
            await self.session.initialize()
            
            # Load tools
            tools_list = await self.session.list_tools()
            self.available_tools = tools_list.tools
            
            # Convert to Gemini format
            gemini_tools = [self._build_function_declaration(tool) for tool in self.available_tools]
            
            # Create model
            self.model = genai.GenerativeModel(
                'models/gemini-2.0-flash-exp',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )
            
            self.initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize GitHub agent: {e}. "
                               "Ensure npx is installed and node.js is up to date. "
                               "The @modelcontextprotocol/server-github package might need authentication on first run.")
    
    async def get_capabilities(self) -> List[str]:
        """Return GitHub capabilities"""
        if not self.available_tools:
            return ["GitHub operations (tools not loaded)"]
        
        capabilities = []
        for tool in self.available_tools:
            capabilities.append(tool.description or tool.name)
        
        return capabilities
    
    async def execute(self, instruction: str) -> str:
        """Execute a GitHub task (sub-orchestration loop)"""
        if not self.initialized:
            return self._format_error(Exception("GitHub agent not initialized"))
        
        try:
            chat = self.model.start_chat()
            response = await chat.send_message_async(instruction)
            
            # Handle function calling loop
            max_iterations = 10
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
                tool_args = self._deep_convert_proto_args(function_call.args)
                
                # Call the tool via MCP
                try:
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    
                    result_content = []
                    for content in tool_result.content:
                        if hasattr(content, 'text'):
                            result_content.append(content.text)
                    
                    result_text = "\n".join(result_content)
                    if not result_text:
                        result_text = json.dumps(tool_result.content, default=str)
                    
                    response = await chat.send_message_async(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"result": result_text}
                                )
                            )]
                        )
                    )
                    
                except Exception as e:
                    error_msg = f"Error calling GitHub tool {tool_name}: {str(e)}"
                    response = await chat.send_message_async(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=tool_name,
                                    response={"error": error_msg}
                                )
                            )]
                        )
                    )
                
                iteration += 1
            
            return response.text
            
        except Exception as e:
            return self._format_error(e)
    
    async def cleanup(self):
        """Disconnect from GitHub"""
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except:
                pass
        
        if self.stdio_context:
            try:
                await self.stdio_context.__aexit__(None, None, None)
            except:
                pass
    
    def _build_function_declaration(self, tool: Any) -> protos.FunctionDeclaration:
        parameters_schema = protos.Schema(type_=protos.Type.OBJECT)
        
        if tool.inputSchema:
            schema = tool.inputSchema
            if "properties" in schema:
                for prop_name, prop_schema in schema["properties"].items():
                    parameters_schema.properties[prop_name] = self._clean_schema(prop_schema)
            
            if "required" in schema:
                parameters_schema.required.extend(schema["required"])
        
        return protos.FunctionDeclaration(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters_schema
        )
    
    def _clean_schema(self, schema: Dict) -> protos.Schema:
        schema_pb = protos.Schema()
        if "type" in schema:
            schema_pb.type_ = self.schema_type_map.get(schema["type"], protos.Type.TYPE_UNSPECIFIED)
        if "description" in schema:
            schema_pb.description = schema["description"]
        if "enum" in schema:
            schema_pb.enum.extend(schema["enum"])
        if "items" in schema and isinstance(schema["items"], dict):
            schema_pb.items = self._clean_schema(schema["items"])
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop_name, prop_schema in schema["properties"].items():
                schema_pb.properties[prop_name] = self._clean_schema(prop_schema)
        if "required" in schema:
            schema_pb.required.extend(schema["required"])
        return schema_pb
    
    def _deep_convert_proto_args(self, value: Any) -> Any:
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value

