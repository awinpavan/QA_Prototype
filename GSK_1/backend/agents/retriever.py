"""
Professional Multi-Agent Data Retriever using LangChain and Groq LLM
This agent intelligently decides which data sources to query based on user input
and provides curated, formatted responses.
"""

import os
import sys
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser

# Groq imports - using native Groq SDK for better performance
from groq import Groq

# Import our custom services
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from services.SOP_GMP_retriver import SOPGMPRetriever
from services.deviation_detector import find_all_deviations_in_directory

# Load environment variables
load_dotenv()

class CustomGroqAgent:
    """
    Custom agent that uses native Groq SDK for better performance and features
    """
    
    def __init__(self, groq_client: Groq, tools: List[Tool], system_prompt: str):
        self.groq_client = groq_client
        self.tools = tools
        self.system_prompt = system_prompt
        self.tool_map = {tool.name: tool for tool in tools}
    
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input and return structured response using Groq SDK
        
        Args:
            input_data: Dictionary containing 'input' key with user query
            
        Returns:
            Dict containing 'output' key with agent response
        """
        user_input = input_data.get("input", "")
        
        # Create messages for Groq API
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # First, get the LLM's decision on which tools to use
        tool_selection_prompt = f"""
Based on the user query: "{user_input}"

Which tools should I use? Respond with a JSON object containing:
{{
    "tools_to_use": ["tool_name1", "tool_name2"],
    "reasoning": "Brief explanation of why these tools are needed"
}}

Available tools:
{self._format_tools_for_prompt()}
"""
        
        messages.append({"role": "user", "content": tool_selection_prompt})
        
        # Get tool selection from Groq
        try:
            tool_selection_response = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_completion_tokens=512
            )
            
            tool_selection = tool_selection_response.choices[0].message.content
            
            # Parse tool selection and execute tools
            tool_results = self._execute_tools(user_input, tool_selection)
            
            # Generate final response with tool results
            final_response = self._generate_final_response(user_input, tool_results)
            
            return {"output": final_response}
            
        except Exception as e:
            return {"output": f"Error processing query: {e}"}
    
    def _format_tools_for_prompt(self) -> str:
        """Format tools for inclusion in prompt"""
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(tool_descriptions)
    
    def _execute_tools(self, user_input: str, tool_selection: str) -> Dict[str, str]:
        """Execute the selected tools and return results"""
        results = {}
        
        try:
            # Simple parsing - look for tool names in the selection
            for tool in self.tools:
                if tool.name in tool_selection.lower():
                    try:
                        result = tool.func(user_input)
                        results[tool.name] = result
                    except Exception as e:
                        results[tool.name] = f"Error executing {tool.name}: {e}"
        except Exception as e:
            results["error"] = f"Error parsing tool selection: {e}"
        
        return results
    
    def _generate_final_response(self, user_input: str, tool_results: Dict[str, str]) -> str:
        """Generate final structured response using Groq"""
        
        # Create prompt for final response generation
        final_prompt = f"""
Based on the user query: "{user_input}"

And the tool results:
{self._format_tool_results(tool_results)}

Provide a structured JSON response in this exact format:
{{
    "SOP": "Relevant SOP/GMP information or null",
    "GMP": "Relevant GMP information or null",
    "Batch_data": "Relevant batch analysis information or null"
}}

Only include data that is directly relevant to the user's query. If no relevant data is found for a category, set it to null.
"""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": final_prompt}
        ]
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_completion_tokens=1024
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating final response: {e}"
    
    def _format_tool_results(self, tool_results: Dict[str, str]) -> str:
        """Format tool results for inclusion in final prompt"""
        formatted = []
        for tool_name, result in tool_results.items():
            formatted.append(f"{tool_name}: {result}")
        return "\n".join(formatted)

@dataclass
class DataRetrieved:
    """Structured output format for retrieved data"""
    SOP: Optional[str] = None
    GMP: Optional[str] = None
    Batch_data: Optional[str] = None

class DataRetrievalAgent:
    """
    Professional multi-agent data retriever that intelligently queries different
    data sources based on user input using LangChain and Groq LLM.
    """
    
    def __init__(self):
        """Initialize the agent with Groq LLM and tools"""
        self.groq_client = self._initialize_groq_client()
        self.sop_retriever = self._initialize_sop_retriever()
        self.tools = self._create_tools()
        self.agent = self._create_agent()
        self.output_parser = JsonOutputParser()
        
    def _initialize_groq_client(self) -> Groq:
        """Initialize Groq client with configuration"""
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            return Groq(api_key=groq_api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Groq client: {e}")
    
    def _call_groq_llm(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Call Groq LLM using native SDK with LangChain-compatible interface
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters for the completion
            
        Returns:
            str: The generated response content
        """
        try:
            # Set default parameters based on Groq documentation
            completion_params = {
                "model": "llama-3.3-70b-versatile",  # Using the recommended model from docs
                "temperature": 0.1,  # Low temperature for consistent decision making
                "max_completion_tokens": 2048,
                "top_p": 1,
                "stream": False,
                **kwargs
            }
            
            # Create chat completion using native Groq SDK
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                **completion_params
            )
            
            return chat_completion.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error calling Groq LLM: {e}")
    
    def _initialize_sop_retriever(self) -> SOPGMPRetriever:
        """Initialize SOP GMP retriever"""
        try:
            return SOPGMPRetriever()
        except Exception as e:
            print(f"Warning: Could not initialize SOP retriever: {e}")
            return None
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools from our existing service files"""
        tools = []
        
        # SOP/GMP Retrieval Tool - using existing SOP_GMP_retriever service
        def sop_gmp_search(query: str) -> str:
            """Search SOP and GMP documents for relevant information"""
            try:
                if not self.sop_retriever:
                    return "SOP/GMP retriever not available"
                
                results = self.sop_retriever.query_sops(query, top_k=3)
                if not results:
                    return "No SOP/GMP documents found for the query"
                
                formatted_results = []
                for result in results:
                    formatted_results.append({
                        "source": result["source_document"],
                        "score": result["score"],
                        "content": result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"]
                    })
                
                return f"SOP/GMP Results: {formatted_results}"
            except Exception as e:
                return f"Error searching SOP/GMP documents: {e}"
        
        # Batch Data Analysis Tool - using existing deviation_detector service
        def batch_data_analysis(query: str) -> str:
            """Analyze batch data for deviations and anomalies"""
            try:
                # Get the path to synthetic batches
                current_dir = os.path.dirname(os.path.abspath(__file__))
                batches_path = os.path.join(current_dir, "..", "..", "data", "synthetic_batches")
                
                if not os.path.exists(batches_path):
                    return "Batch data directory not found"
                
                # Use the existing deviation_detector service function
                deviations = find_all_deviations_in_directory(batches_path)
                
                if not deviations:
                    return "No deviations found in batch data"
                
                # Group deviations by type for better analysis
                deviation_summary = {}
                for dev in deviations:
                    reason = dev.get('deviation_reason', 'Unknown')
                    if reason not in deviation_summary:
                        deviation_summary[reason] = 0
                    deviation_summary[reason] += 1
                
                # Also include some sample deviations for context
                sample_deviations = deviations[:3] if deviations else []
                
                return f"Batch Analysis Results - Summary: {deviation_summary}, Sample deviations: {sample_deviations}"
            except Exception as e:
                return f"Error analyzing batch data: {e}"
        
        # Add tools to the list
        tools.extend([
            Tool(
                name="sop_gmp_search",
                description="Search SOP (Standard Operating Procedures) and GMP (Good Manufacturing Practices) documents for compliance information, procedures, and guidelines. Use this for questions about protocols, procedures, guidelines, compliance, SOPs, GMPs.",
                func=sop_gmp_search
            ),
            Tool(
                name="batch_data_analysis", 
                description="Analyze batch manufacturing data for deviations, anomalies, and compliance issues. Use this for questions about manufacturing data, batch analysis, deviations, anomalies, sensor failures, calibration issues.",
                func=batch_data_analysis
            )
        ])
        
        return tools
    
    def _create_agent(self) -> 'CustomGroqAgent':
        """Create a custom agent that uses native Groq SDK"""
        return CustomGroqAgent(
            groq_client=self.groq_client,
            tools=self.tools,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return """You are a professional pharmaceutical data retrieval agent powered by Groq LLM. Your job is to intelligently determine which data sources to query based on the user's request and provide a structured response.

Available tools:
1. sop_gmp_search: Searches SOP (Standard Operating Procedures) and GMP (Good Manufacturing Practices) documents using Cohere embeddings and Pinecone vector database. Use for questions about procedures, compliance, guidelines, protocols, temperature deviations, pressure anomalies, microbial contamination, etc.
2. batch_data_analysis: Analyzes batch manufacturing data for deviations and anomalies using the deviation_detector service. Use for questions about manufacturing data, batch analysis, deviations, anomalies, sensor failures, calibration issues, etc.

Instructions:
- Analyze the user's query carefully to determine what type of data they need
- Only use tools that are relevant to their request
- If they ask about procedures, guidelines, compliance, SOPs, GMPs → use sop_gmp_search
- If they ask about manufacturing data, deviations, anomalies, batch analysis → use batch_data_analysis  
- If they ask about both types of data, use both tools
- Always provide a structured JSON response in the following format:

{
    "SOP": "Relevant SOP/GMP information from the search results or null if not requested",
    "GMP": "Relevant GMP information or null if not requested", 
    "Batch_data": "Relevant batch analysis information or null if not requested"
}

Be concise but informative. Only include data that is directly relevant to the user's query. If no relevant data is found, set the appropriate field to null."""
    
    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Main method to process user queries and return structured data using native Groq SDK
        
        Args:
            user_input (str): The user's query
            
        Returns:
            Dict containing structured data with SOP, GMP, and Batch_data fields
        """
        try:
            # Get response from the custom agent
            response = self.agent.invoke({"input": user_input})
            
            # Extract the actual content from the agent response
            agent_output = response.get("output", "")
            
            # Try to parse the response as JSON using native Groq SDK
            try:
                import json
                import re
                
                # Try to find JSON pattern in the response - look for the complete JSON structure
                json_pattern = r'\{[^{}]*"SOP"[^{}]*"GMP"[^{}]*"Batch_data"[^{}]*\}'
                json_match = re.search(json_pattern, agent_output, re.DOTALL)
                
                if json_match:
                    try:
                        parsed_data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        # Try a more flexible pattern
                        flexible_pattern = r'\{[^{}]*"SOP"[^{}]*\}'
                        flexible_match = re.search(flexible_pattern, agent_output, re.DOTALL)
                        if flexible_match:
                            parsed_data = json.loads(flexible_match.group())
                        else:
                            parsed_data = self._parse_unstructured_response(agent_output, user_input)
                else:
                    # Fallback: create structured response based on content
                    parsed_data = self._parse_unstructured_response(agent_output, user_input)
                
                return {
                    "data_retrieved": DataRetrieved(
                        SOP=parsed_data.get("SOP"),
                        GMP=parsed_data.get("GMP"), 
                        Batch_data=parsed_data.get("Batch_data")
                    ),
                    "raw_response": agent_output,
                    "status": "success"
                }
                
            except Exception as parse_error:
                # Fallback parsing
                parsed_data = self._parse_unstructured_response(agent_output, user_input)
                return {
                    "data_retrieved": DataRetrieved(
                        SOP=parsed_data.get("SOP"),
                        GMP=parsed_data.get("GMP"),
                        Batch_data=parsed_data.get("Batch_data")
                    ),
                    "raw_response": agent_output,
                    "status": "success_with_fallback_parsing"
                }
                
        except Exception as e:
            return {
                "data_retrieved": DataRetrieved(),
                "error": str(e),
                "status": "error"
            }
    
    def _parse_unstructured_response(self, response: str, user_input: str) -> Dict[str, str]:
        """Parse unstructured response and categorize content based on keywords and context"""
        parsed = {"SOP": None, "GMP": None, "Batch_data": None}
        
        user_lower = user_input.lower()
        response_lower = response.lower()
        
        # Check if response contains SOP/GMP search results
        if "sop/gmp results" in response_lower or "source_document" in response_lower:
            # This looks like SOP/GMP search results
            if "gmp" in response_lower and "sop" in response_lower:
                # Contains both, try to separate or put in SOP
                parsed["SOP"] = response
            elif "gmp" in response_lower:
                parsed["GMP"] = response
            else:
                parsed["SOP"] = response
        
        # Check if response contains batch analysis results
        elif "batch analysis results" in response_lower or "deviation" in response_lower:
            parsed["Batch_data"] = response
        
        # Fallback to keyword-based categorization
        else:
            # Check for SOP/GMP related keywords in user input
            sop_gmp_keywords = ["sop", "procedure", "guideline", "compliance", "gmp", "protocol", 
                              "temperature deviation", "pressure anomaly", "microbial contamination"]
            batch_keywords = ["batch", "manufacturing", "deviation", "anomaly", "data", "sensor", 
                            "calibration", "failure"]
            
            if any(keyword in user_lower for keyword in sop_gmp_keywords):
                if "gmp" in response_lower:
                    parsed["GMP"] = response
                else:
                    parsed["SOP"] = response
            
            if any(keyword in user_lower for keyword in batch_keywords):
                parsed["Batch_data"] = response
        
        return parsed
    
    def format_output(self, result: Dict[str, Any]) -> str:
        """Format the final output in the requested format"""
        data = result.get("data_retrieved", DataRetrieved())
        
        output = "data_retrieved: "
        output += f"SOP = {data.SOP if data.SOP else 'null'}\n"
        output += f"                   GMP = {data.GMP if data.GMP else 'null'}\n"
        output += f"                   Batch_data = {data.Batch_data if data.Batch_data else 'null'}"
        
        return output

# Interactive interface
def main():
    """Interactive command-line interface for the data retrieval agent"""
    print("🤖 Professional Multi-Agent Data Retriever")
    print("=" * 60)
    print("Powered by Native Groq SDK + LangChain Tools")
    print("Available data sources: SOP/GMP documents, Batch manufacturing data")
    print("=" * 60)
    
    try:
        agent = DataRetrievalAgent()
        print("✅ Agent initialized successfully!")
        
        while True:
            try:
                user_input = input("\n📝 Enter your query (or 'quit' to exit): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("❌ Please enter a query.")
                    continue
                
                print("\n🔍 Processing your query...")
                result = agent.query(user_input)
                
                print("\n" + "="*60)
                print("📊 RESULTS:")
                print("="*60)
                print(agent.format_output(result))
                
                if result.get("status") == "error":
                    print(f"\n❌ Error: {result.get('error')}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")

if __name__ == "__main__":
    main()
