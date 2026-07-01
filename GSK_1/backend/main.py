"""
Multi-Agent Orchestration System using LangGraph
This file orchestrates the planner and retriever agents using LangGraph workflow management.
"""

import os
import sys
from typing import Dict, Any, List, TypedDict, Annotated
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# Import our custom agents
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))
from planner import GroqPlannerAgent, ExecutionPlan, PlanStep
from retriever import DataRetrievalAgent

# Load environment variables
load_dotenv()

class AgentState(TypedDict):
    """State management for the multi-agent workflow"""
    # Input
    user_query: str
    
    # Planning phase
    execution_plan: ExecutionPlan
    plan_summary: Dict[str, Any]
    
    # Retrieval phase
    retrieval_results: List[Dict[str, Any]]
    
    # Output preparation
    consolidated_data: Dict[str, Any]
    final_response: str
    
    # Workflow control
    current_step: int
    completed_steps: List[int]
    errors: List[str]
    status: str  # "planning", "retrieving", "completed", "error"

class MultiAgentOrchestrator:
    """
    Multi-agent orchestrator using LangGraph for workflow management
    """
    
    def __init__(self):
        """Initialize the orchestrator with agents"""
        self.planner = self._initialize_planner()
        self.retriever = self._initialize_retriever()
        self.workflow = self._create_workflow()
        
    def _initialize_planner(self) -> GroqPlannerAgent:
        """Initialize the planner agent"""
        try:
            return GroqPlannerAgent()
        except Exception as e:
            raise Exception(f"Failed to initialize planner agent: {e}")
    
    def _initialize_retriever(self) -> DataRetrievalAgent:
        """Initialize the retriever agent"""
        try:
            return DataRetrievalAgent()
        except Exception as e:
            raise Exception(f"Failed to initialize retriever agent: {e}")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("retriever", self._retriever_node)
        workflow.add_node("consolidator", self._consolidator_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges
        workflow.add_edge("planner", "retriever")
        workflow.add_edge("retriever", "consolidator")
        workflow.add_edge("consolidator", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def _planner_node(self, state: AgentState) -> AgentState:
        """
        Planner node: Creates execution plan for the user query
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with execution plan
        """
        try:
            print(f"\n🎯 PLANNER NODE: Creating execution plan for '{state['user_query']}'")
            
            # Create execution plan using the planner agent
            execution_plan = self.planner.create_execution_plan(state["user_query"])
            plan_summary = self.planner.get_plan_summary(execution_plan)
            
            print(f"✅ Plan created: {execution_plan.plan_id}")
            print(f"📊 Steps: {len(execution_plan.steps)}")
            print(f"🤖 Required agents: {', '.join(execution_plan.required_agents)}")
            
            # Update state
            state["execution_plan"] = execution_plan
            state["plan_summary"] = plan_summary
            state["status"] = "retrieving"
            state["current_step"] = 0
            state["completed_steps"] = []
            state["retrieval_results"] = []
            
            return state
            
        except Exception as e:
            error_msg = f"Planner node error: {e}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["status"] = "error"
            return state
    
    def _retriever_node(self, state: AgentState) -> AgentState:
        """
        Retriever node: Executes retrieval steps from the plan
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with retrieval results
        """
        try:
            print(f"\n🔍 RETRIEVER NODE: Executing retrieval steps")
            
            execution_plan = state["execution_plan"]
            retrieval_results = []
            
            # Execute each step that requires the retriever agent
            for step in execution_plan.steps:
                if step.agent_type == "retriever":
                    print(f"  📋 Executing step {step.step_id}: {step.action}")
                    
                    try:
                        # Execute the retriever based on the step action
                        if step.action == "search_sop_gmp_documents":
                            query_text = step.input_parameters.get("query_text", state["user_query"])
                            top_k = step.input_parameters.get("top_k", 5)
                            
                            result = self.retriever.query(query_text)
                            retrieval_results.append({
                                "step_id": step.step_id,
                                "action": step.action,
                                "result": result,
                                "status": "success"
                            })
                            
                            print(f"    ✅ SOP/GMP search completed")
                            
                        elif step.action == "analyze_batch_data":
                            query_text = step.input_parameters.get("query_text", state["user_query"])
                            
                            result = self.retriever.query(query_text)
                            retrieval_results.append({
                                "step_id": step.step_id,
                                "action": step.action,
                                "result": result,
                                "status": "success"
                            })
                            
                            print(f"    ✅ Batch data analysis completed")
                        
                        state["completed_steps"].append(step.step_id)
                        
                    except Exception as step_error:
                        error_msg = f"Step {step.step_id} execution error: {step_error}"
                        print(f"    ❌ {error_msg}")
                        retrieval_results.append({
                            "step_id": step.step_id,
                            "action": step.action,
                            "result": None,
                            "status": "error",
                            "error": str(step_error)
                        })
                        state["errors"].append(error_msg)
            
            # Update state
            state["retrieval_results"] = retrieval_results
            state["status"] = "consolidating"
            
            print(f"✅ Retrieval phase completed: {len(retrieval_results)} results")
            
            return state
            
        except Exception as e:
            error_msg = f"Retriever node error: {e}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["status"] = "error"
            return state
    
    def _consolidator_node(self, state: AgentState) -> AgentState:
        """
        Consolidator node: Prepares data for the executor agent (next step)
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with consolidated data
        """
        try:
            print(f"\n📊 CONSOLIDATOR NODE: Preparing data for executor")
            
            # Consolidate retrieval results
            consolidated_data = {
                "user_query": state["user_query"],
                "execution_plan": state["execution_plan"],
                "retrieval_results": state["retrieval_results"],
                "plan_summary": state["plan_summary"]
            }
            
            # Prepare structured output for executor
            structured_output = {
                "SOP": None,
                "GMP": None,
                "Batch_data": None
            }
            
            # Extract data from retrieval results
            for result in state["retrieval_results"]:
                if result["status"] == "success" and result["result"]:
                    data_retrieved = result["result"].get("data_retrieved")
                    if data_retrieved:
                        if data_retrieved.SOP:
                            structured_output["SOP"] = data_retrieved.SOP
                        if data_retrieved.GMP:
                            structured_output["GMP"] = data_retrieved.GMP
                        if data_retrieved.Batch_data:
                            structured_output["Batch_data"] = data_retrieved.Batch_data
            
            consolidated_data["structured_output"] = structured_output
            
            # Create a preliminary response
            preliminary_response = self._create_preliminary_response(consolidated_data)
            
            # Update state
            state["consolidated_data"] = consolidated_data
            state["final_response"] = preliminary_response
            state["status"] = "completed"
            
            print(f"✅ Consolidation completed")
            print(f"📋 Data ready for executor agent")
            
            return state
            
        except Exception as e:
            error_msg = f"Consolidator node error: {e}"
            print(f"❌ {error_msg}")
            state["errors"].append(error_msg)
            state["status"] = "error"
            return state
    
    def _create_preliminary_response(self, consolidated_data: Dict[str, Any]) -> str:
        """Create a preliminary response from consolidated data"""
        
        structured_output = consolidated_data.get("structured_output", {})
        
        response_parts = []
        
        if structured_output.get("SOP"):
            response_parts.append(f"📋 SOP Information:\n{structured_output['SOP']}")
        
        if structured_output.get("GMP"):
            response_parts.append(f"📋 GMP Information:\n{structured_output['GMP']}")
        
        if structured_output.get("Batch_data"):
            response_parts.append(f"📊 Batch Data Analysis:\n{structured_output['Batch_data']}")
        
        if not response_parts:
            response_parts.append("No relevant data found for the query.")
        
        return "\n\n".join(response_parts)
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Main method to process user queries through the multi-agent workflow
        
        Args:
            user_query (str): The user's query
            
        Returns:
            Dict containing the complete workflow results
        """
        try:
            print(f"\n🚀 Starting Multi-Agent Workflow")
            print(f"📝 Query: {user_query}")
            print("=" * 60)
            
            # Initialize state
            initial_state: AgentState = {
                "user_query": user_query,
                "execution_plan": None,
                "plan_summary": {},
                "retrieval_results": [],
                "consolidated_data": {},
                "final_response": "",
                "current_step": 0,
                "completed_steps": [],
                "errors": [],
                "status": "planning"
            }
            
            # Execute the workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Prepare results
            results = {
                "status": final_state["status"],
                "user_query": user_query,
                "execution_plan": final_state.get("execution_plan"),
                "plan_summary": final_state.get("plan_summary", {}),
                "retrieval_results": final_state.get("retrieval_results", []),
                "consolidated_data": final_state.get("consolidated_data", {}),
                "final_response": final_state.get("final_response", ""),
                "errors": final_state.get("errors", []),
                "completed_steps": final_state.get("completed_steps", [])
            }
            
            print(f"\n✅ Workflow completed with status: {final_state['status']}")
            
            return results
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "user_query": user_query,
                "execution_plan": None,
                "plan_summary": {},
                "retrieval_results": [],
                "consolidated_data": {},
                "final_response": "",
                "errors": [str(e)],
                "completed_steps": []
            }
    
    def format_output(self, results: Dict[str, Any]) -> str:
        """Format the workflow results for display"""
        
        if results["status"] == "error":
            return f"❌ Error: {results.get('error', 'Unknown error')}"
        
        output_parts = []
        
        # Add plan summary
        plan_summary = results.get("plan_summary", {})
        if plan_summary:
            output_parts.append(f"🎯 Plan: {plan_summary.get('objective', 'N/A')}")
            output_parts.append(f"⚡ Complexity: {plan_summary.get('complexity_level', 'N/A')}")
            output_parts.append(f"📊 Steps: {plan_summary.get('total_steps', 0)}")
        
        # Add final response
        final_response = results.get("final_response", "")
        if final_response:
            output_parts.append(f"\n📋 RESULTS:\n{final_response}")
        
        # Add errors if any
        errors = results.get("errors", [])
        if errors:
            output_parts.append(f"\n⚠️ Warnings: {len(errors)} issues encountered")
        
        return "\n".join(output_parts) if output_parts else "No results generated"

# Interactive interface
def main():
    """Interactive command-line interface for the multi-agent orchestrator"""
    print("🤖 Multi-Agent Orchestration System")
    print("=" * 60)
    print("Powered by LangGraph + Groq LLM")
    print("Agents: Planner → Retriever → Consolidator")
    print("=" * 60)
    
    try:
        orchestrator = MultiAgentOrchestrator()
        print("✅ Multi-agent orchestrator initialized successfully!")
        
        while True:
            try:
                user_input = input("\n📝 Enter your query (or 'quit' to exit): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("❌ Please enter a query.")
                    continue
                
                # Process the query through the workflow
                results = orchestrator.process_query(user_input)
                
                print("\n" + "="*60)
                print("📊 WORKFLOW RESULTS:")
                print("="*60)
                print(orchestrator.format_output(results))
                
                # Show detailed results for debugging
                print(f"\n🔍 DETAILED STATUS:")
                print(f"   Status: {results['status']}")
                print(f"   Completed Steps: {results['completed_steps']}")
                print(f"   Errors: {len(results['errors'])}")
                
                if results.get("consolidated_data"):
                    print(f"\n📋 READY FOR EXECUTOR AGENT:")
                    structured_output = results["consolidated_data"].get("structured_output", {})
                    print(f"   SOP Data: {'✅' if structured_output.get('SOP') else '❌'}")
                    print(f"   GMP Data: {'✅' if structured_output.get('GMP') else '❌'}")
                    print(f"   Batch Data: {'✅' if structured_output.get('Batch_data') else '❌'}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")

if __name__ == "__main__":
    main()

