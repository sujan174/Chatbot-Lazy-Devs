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
from connectors.base_agent import BaseAgent


class Agent(BaseAgent):
    """Specialized agent for Slack operations"""
    
    def __init__(self, user_id: str):
        super().__init__(user_id) # Pass user_id to BaseAgent
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
        You are a Slack specialist agent. You have access to Slack tools to:
        - Send messages to channels and users
        - Read channel history
        - List channels and users
        - Add reactions to messages
        - Search messages
        
        Always:
        1. Confirm channel names before sending messages
        2. Format messages clearly and professionally
        3. Provide summaries of what was sent or found
        4. Handle errors gracefully
        5. Respect Slack etiquette (don't spam, be concise)
        """
    
    async def initialize(self):
        """Connect to Slack MCP server"""
        try:
            # NOTE: This is where you would fetch user-specific credentials
            # using self.user_id from a secure vault.
            # In a real app, you'd do:
            # creds = await get_slack_creds_for_user(self.user_id)
            # slack_bot_token = creds['bot_token']
            # slack_team_id = creds['team_id']
            
            slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
            slack_team_id = os.environ.get("SLACK_TEAM_ID")
            
            if not slack_bot_token or not slack_team_id:
                raise ValueError(f"SLACK_BOT_TOKEN and SLACK_TEAM_ID must be set for user {self.user_id}")
            
            server_params = StdioServerParameters(
                command="npx",
                args=["-y", "@modelcontextprotocol/server-slack"],
                env={
                    **os.environ,
                    "SLACK_BOT_TOKEN": slack_bot_token,
                    "SLACK_TEAM_ID": slack_team_id
                }
            )
            
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
                'models/gemini-2.5-flash',
                system_instruction=self.system_prompt,
                tools=gemini_tools
            )
            
            self.initialized = True
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Slack agent for user {self.user_id}: {e}")
    
    async def get_capabilities(self) -> List[str]:
        """Return Slack capabilities"""
        if not self.available_tools:
            return ["Slack operations (tools not loaded)"]
        
        capabilities = []
        for tool in self.available_tools:
            capabilities.append(tool.description or tool.name)
        
        return capabilities
    
    async def execute(self, instruction: str) -> str:
        """Execute a Slack task"""
        if not self.initialized:
            return "Error: Slack agent not initialized"
        
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
                
                # Get function call
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
                    
                    # Send result back
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
                    error_msg = f"Error calling Slack tool {tool_name}: {str(e)}"
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
        """Disconnect from Slack"""
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
        """Convert MCP tool to Gemini function declaration"""
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
        """Convert JSON schema to protobuf schema"""
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
        """Convert protobuf types to Python types"""
        type_str = str(type(value))
        if "MapComposite" in type_str:
            return {k: self._deep_convert_proto_args(v) for k, v in value.items()}
        elif "RepeatedComposite" in type_str:
            return [self._deep_convert_proto_args(item) for item in value]
        else:
            return value

