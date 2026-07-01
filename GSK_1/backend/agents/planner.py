"""
Professional Multi-Agent Planner using LangChain and Groq LLM
This agent serves as the orchestrator that receives user queries and breaks them down
into actionable plans for other agents in the system.
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from dotenv import load_dotenv

# Groq imports - using native Groq SDK for better performance
from groq import Groq

# Load environment variables
load_dotenv()

@dataclass
class PlanStep:
    """Represents a single step in the execution plan"""
    step_id: int
    agent_type: str  # "retriever", "analyzer", "reporter", etc.
    action: str
    description: str
    input_parameters: Dict[str, Any]
    expected_output: str
    dependencies: List[int] = None  # Step IDs this step depends on
    priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class ExecutionPlan:
    """Complete execution plan for a user query"""
    plan_id: str
    user_query: str
    objective: str
    steps: List[PlanStep]
    estimated_duration: str
    complexity_level: str  # "simple", "moderate", "complex"
    required_agents: List[str]

class GroqPlannerAgent:
    """
    Professional planner agent that creates execution plans for multi-agent workflows
    using Groq LLM for intelligent planning and decision making.
    This agent only focuses on planning - execution will be handled by LangGraph orchestration.
    """
    
    def __init__(self):
        """Initialize the planner agent with Groq LLM"""
        self.groq_client = self._initialize_groq_client()
        self.system_prompt = self._get_system_prompt()
        
    def _initialize_groq_client(self) -> Groq:
        """Initialize Groq client with configuration"""
        try:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            return Groq(api_key=groq_api_key)
        except Exception as e:
            raise Exception(f"Failed to initialize Groq client: {e}")
    
    def get_available_agents_info(self) -> Dict[str, str]:
        """Get information about available agents in the system"""
        return {
            "retriever": "Searches SOP/GMP documents using Cohere embeddings and Pinecone vector database, analyzes batch manufacturing data for deviations and anomalies",
            "analyzer": "Analyzes data patterns and generates insights (planned)",
            "reporter": "Generates formatted reports and summaries (planned)"
        }
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the planner agent"""
        return """You are a professional multi-agent orchestrator and planner. Your role is to:

1. ANALYZE user queries to understand their intent and requirements
2. BREAK DOWN complex queries into actionable execution plans
3. ORCHESTRATE multiple agents to fulfill the user's needs
4. COORDINATE data flow between agents
5. ENSURE comprehensive and accurate responses

Available Agents:
- retriever: Retrieves SOP/GMP documents and batch manufacturing data using Cohere embeddings and Pinecone vector database

Planning Guidelines:
- Start with understanding the user's primary objective
- Identify what data sources and agents are needed
- Create a logical sequence of steps
- Consider dependencies between steps
- Estimate complexity and duration
- Ensure all aspects of the query are addressed

Always provide structured plans with clear steps, dependencies, and expected outcomes."""

    def create_execution_plan(self, user_query: str) -> ExecutionPlan:
        """
        Create an execution plan for the user query using Groq LLM
        
        Args:
            user_query (str): The user's query or request
            
        Returns:
            ExecutionPlan: Complete execution plan with steps and orchestration details
        """
        try:
            # Generate plan using Groq LLM
            plan_json = self._generate_plan_with_groq(user_query)
            
            # Parse and create ExecutionPlan object
            execution_plan = self._parse_plan_json(plan_json, user_query)
            
            return execution_plan
            
        except Exception as e:
            # Fallback plan creation
            return self._create_fallback_plan(user_query, str(e))
    
    def _generate_plan_with_groq(self, user_query: str) -> str:
        """Generate execution plan using Groq LLM"""
        
        planning_prompt = f"""
User Query: "{user_query}"

Available Agents:
{self._format_available_agents()}

Create a detailed execution plan in the following JSON format:
{{
    "objective": "Clear description of what we're trying to achieve",
    "complexity_level": "simple|moderate|complex",
    "estimated_duration": "e.g., '2-3 minutes'",
    "required_agents": ["list", "of", "required", "agents"],
    "steps": [
        {{
            "step_id": 1,
            "agent_type": "retriever",
            "action": "search_sop_gmp_documents",
            "description": "Search for SOP and GMP documents related to the query",
            "input_parameters": {{
                "query_text": "extracted or refined query text",
                "top_k": 5
            }},
            "expected_output": "Relevant SOP/GMP documents and procedures",
            "dependencies": [],
            "priority": 1
        }},
        {{
            "step_id": 2,
            "agent_type": "retriever", 
            "action": "analyze_batch_data",
            "description": "Analyze batch manufacturing data for deviations",
            "input_parameters": {{
                "query_text": "batch analysis query"
            }},
            "expected_output": "Batch data analysis results and deviations",
            "dependencies": [],
            "priority": 2
        }}
    ]
}}

Guidelines:
- Only include steps that are relevant to the user query
- Use appropriate agent types and actions
- Set realistic priorities and dependencies
- Provide clear descriptions and expected outputs
- Consider the complexity of the request
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": planning_prompt}
        ]
        
        try:
            response = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                temperature=0.1,  # Low temperature for consistent planning
                max_completion_tokens=2048
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Error generating plan with Groq: {e}")
    
    def _format_available_agents(self) -> str:
        """Format available agents for the planning prompt"""
        agent_descriptions = []
        agents_info = self.get_available_agents_info()
        
        agent_descriptions.append("""
- retriever: 
  * Actions: search_sop_gmp_documents, analyze_batch_data
  * Capabilities: Searches SOP/GMP documents using Cohere embeddings and Pinecone vector database, analyzes batch manufacturing data for deviations and anomalies
  * Input: query_text, top_k (optional)
  * Output: Structured data with SOP, GMP, and Batch_data fields
""")
        
        # Add other agents as they become available
        for agent_name, description in agents_info.items():
            if agent_name != "retriever":
                agent_descriptions.append(f"""
- {agent_name}: {description}
""")
        
        return "\n".join(agent_descriptions)
    
    def _parse_plan_json(self, plan_json: str, user_query: str) -> ExecutionPlan:
        """Parse JSON plan and create ExecutionPlan object"""
        try:
            import re
            
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', plan_json, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in plan response")
            
            # Create PlanStep objects
            steps = []
            for step_data in plan_data.get("steps", []):
                step = PlanStep(
                    step_id=step_data.get("step_id", 0),
                    agent_type=step_data.get("agent_type", ""),
                    action=step_data.get("action", ""),
                    description=step_data.get("description", ""),
                    input_parameters=step_data.get("input_parameters", {}),
                    expected_output=step_data.get("expected_output", ""),
                    dependencies=step_data.get("dependencies", []),
                    priority=step_data.get("priority", 1)
                )
                steps.append(step)
            
            # Create ExecutionPlan object
            execution_plan = ExecutionPlan(
                plan_id=f"plan_{hash(user_query) % 10000}",
                user_query=user_query,
                objective=plan_data.get("objective", "Process user query"),
                steps=steps,
                estimated_duration=plan_data.get("estimated_duration", "Unknown"),
                complexity_level=plan_data.get("complexity_level", "simple"),
                required_agents=plan_data.get("required_agents", [])
            )
            
            return execution_plan
            
        except Exception as e:
            raise Exception(f"Error parsing plan JSON: {e}")
    
    def _create_fallback_plan(self, user_query: str, error_msg: str) -> ExecutionPlan:
        """Create a fallback plan when Groq planning fails"""
        
        # Simple heuristic-based planning
        query_lower = user_query.lower()
        
        steps = []
        required_agents = []
        
        # Check if SOP/GMP search is needed
        sop_keywords = ["sop", "procedure", "guideline", "compliance", "gmp", "protocol", 
                       "temperature", "pressure", "microbial", "contamination"]
        if any(keyword in query_lower for keyword in sop_keywords):
            steps.append(PlanStep(
                step_id=1,
                agent_type="retriever",
                action="search_sop_gmp_documents",
                description="Search SOP and GMP documents for relevant information",
                input_parameters={"query_text": user_query, "top_k": 5},
                expected_output="Relevant SOP/GMP documents and procedures",
                dependencies=[],
                priority=1
            ))
            required_agents.append("retriever")
        
        # Check if batch analysis is needed
        batch_keywords = ["batch", "manufacturing", "deviation", "anomaly", "data", 
                         "sensor", "calibration", "failure", "analysis"]
        if any(keyword in query_lower for keyword in batch_keywords):
            steps.append(PlanStep(
                step_id=2,
                agent_type="retriever",
                action="analyze_batch_data",
                description="Analyze batch manufacturing data for deviations",
                input_parameters={"query_text": user_query},
                expected_output="Batch data analysis results and deviations",
                dependencies=[],
                priority=1 if "batch" in query_lower else 2
            ))
            required_agents.append("retriever")
        
        # If no specific keywords found, do a general search
        if not steps:
            steps.append(PlanStep(
                step_id=1,
                agent_type="retriever",
                action="search_sop_gmp_documents",
                description="Perform general search for relevant information",
                input_parameters={"query_text": user_query, "top_k": 5},
                expected_output="General search results",
                dependencies=[],
                priority=1
            ))
            required_agents.append("retriever")
        
        return ExecutionPlan(
            plan_id=f"fallback_plan_{hash(user_query) % 10000}",
            user_query=user_query,
            objective="Process user query with fallback planning",
            steps=steps,
            estimated_duration="1-2 minutes",
            complexity_level="simple",
            required_agents=list(set(required_agents))  # Remove duplicates
        )
    
    def get_plan_summary(self, execution_plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Get a summary of the execution plan for display purposes
        
        Args:
            execution_plan (ExecutionPlan): The execution plan to summarize
            
        Returns:
            Dict containing plan summary information
        """
        return {
            "plan_id": execution_plan.plan_id,
            "user_query": execution_plan.user_query,
            "objective": execution_plan.objective,
            "complexity_level": execution_plan.complexity_level,
            "estimated_duration": execution_plan.estimated_duration,
            "required_agents": execution_plan.required_agents,
            "total_steps": len(execution_plan.steps),
            "steps": [asdict(step) for step in execution_plan.steps]
        }

# Interactive interface
def main():
    """Interactive command-line interface for the planner agent"""
    print("🎯 Professional Multi-Agent Planner")
    print("=" * 60)
    print("Powered by Native Groq SDK + Planning Intelligence")
    print("Available Agents: Retriever (SOP/GMP + Batch Data)")
    print("=" * 60)
    
    try:
        planner = GroqPlannerAgent()
        print("✅ Planner agent initialized successfully!")
        
        while True:
            try:
                user_input = input("\n📝 Enter your query (or 'quit' to exit): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("❌ Please enter a query.")
                    continue
                
                print("\n🎯 Creating execution plan...")
                execution_plan = planner.create_execution_plan(user_input)
                plan_summary = planner.get_plan_summary(execution_plan)
                
                print("\n" + "="*60)
                print("📋 EXECUTION PLAN:")
                print("="*60)
                
                print(f"🎯 Plan ID: {plan_summary['plan_id']}")
                print(f"📝 User Query: {plan_summary['user_query']}")
                print(f"🎯 Objective: {plan_summary['objective']}")
                print(f"⚡ Complexity: {plan_summary['complexity_level']}")
                print(f"⏱️ Estimated Duration: {plan_summary['estimated_duration']}")
                print(f"🤖 Required Agents: {', '.join(plan_summary['required_agents'])}")
                print(f"📊 Total Steps: {plan_summary['total_steps']}")
                
                print(f"\n📋 EXECUTION STEPS:")
                print("-" * 40)
                for step in plan_summary['steps']:
                    print(f"Step {step['step_id']}: {step['action']}")
                    print(f"  Agent: {step['agent_type']}")
                    print(f"  Description: {step['description']}")
                    print(f"  Priority: {step['priority']}")
                    print(f"  Dependencies: {step['dependencies']}")
                    print()
                
                print("✅ Plan created successfully! Ready for LangGraph orchestration.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize planner agent: {e}")

if __name__ == "__main__":
    main()
