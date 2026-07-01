"""
Executor Agent for carrying out planned tasks and operations.
Handles data fetching, n8n workflow execution, and task coordination.
"""

import os
import csv
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys

# Add the services directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

from n8n_client import N8NClient
from deviation_detector import find_deviations

class ExecutorAgent:
    """
    Agent responsible for executing planned tasks and operations.
    Handles data fetching, workflow execution, and result coordination.
    """
    
    def __init__(self):
        """Initialize the executor with required services."""
        self.n8n_client = N8NClient()
        self.tasks_executed = []
        self.results = {}
    
    def fetch_logs(self, source: str = "data/synthetic_batches") -> Dict[str, Any]:
        """
        Fetch batch logs from the specified source.
        
        Args:
            source: Path to batch logs directory or file
            
        Returns:
            Dictionary containing log data and metadata
        """
        task_id = f"fetch_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if os.path.isfile(source):
                # Single file processing
                with open(source, 'r', encoding='utf-8') as f:
                    if source.endswith('.csv'):
                        reader = csv.DictReader(f)
                        log_data = list(reader)
                    else:
                        log_data = f.read()
                
                result = {
                    "task_id": task_id,
                    "status": "completed",
                    "source": source,
                    "data_type": "single_file",
                    "record_count": len(log_data) if isinstance(log_data, list) else 1,
                    "data": log_data
                }
                
            elif os.path.isdir(source):
                # Directory processing
                log_files = []
                total_records = 0
                
                for filename in os.listdir(source):
                    if filename.endswith('.csv'):
                        file_path = os.path.join(source, filename)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            file_data = list(reader)
                            log_files.append({
                                "filename": filename,
                                "path": file_path,
                                "record_count": len(file_data),
                                "data": file_data
                            })
                            total_records += len(file_data)
                
                result = {
                    "task_id": task_id,
                    "status": "completed",
                    "source": source,
                    "data_type": "directory",
                    "file_count": len(log_files),
                    "total_records": total_records,
                    "files": log_files
                }
            else:
                result = {
                    "task_id": task_id,
                    "status": "failed",
                    "error": f"Source '{source}' is neither a file nor a directory",
                    "source": source
                }
            
            # Record the task
            self.tasks_executed.append({
                "task": "fetch_logs",
                "task_id": task_id,
                "status": result["status"],
                "timestamp": datetime.now().isoformat(),
                "source": source
            })
            
            self.results[task_id] = result
            return result
            
        except Exception as e:
            error_result = {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "source": source
            }
            
            self.tasks_executed.append({
                "task": "fetch_logs",
                "task_id": task_id,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "source": source
            })
            
            self.results[task_id] = error_result
            return error_result
    
    def run_deviation_detector(self, batch_source: str) -> Dict[str, Any]:
        """
        Run the deviation detector on batch data.
        
        Args:
            batch_source: Path to batch data (file or directory)
            
        Returns:
            Dictionary containing deviation detection results
        """
        task_id = f"deviation_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Use the deviation detector service
            deviations = find_deviations(batch_source)
            
            result = {
                "task_id": task_id,
                "status": "completed",
                "source": batch_source,
                "deviation_count": len(deviations),
                "deviations": deviations,
                "summary": {
                    "total_deviations": len(deviations),
                    "by_reason": self._group_deviations_by_reason(deviations),
                    "by_file": self._group_deviations_by_file(deviations)
                }
            }
            
            # Record the task
            self.tasks_executed.append({
                "task": "run_deviation_detector",
                "task_id": task_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "source": batch_source,
                "deviation_count": len(deviations)
            })
            
            self.results[task_id] = result
            return result
            
        except Exception as e:
            error_result = {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "source": batch_source
            }
            
            self.tasks_executed.append({
                "task": "run_deviation_detector",
                "task_id": task_id,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "source": batch_source
            })
            
            self.results[task_id] = error_result
            return error_result
    
    def call_n8n_workflow(self, workflow_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an n8n workflow with the given payload.
        
        Args:
            workflow_id: The workflow ID or webhook path
            payload: Data to send to the workflow
            
        Returns:
            Dictionary containing the workflow execution result
        """
        task_id = f"n8n_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Call the n8n workflow
            result = self.n8n_client.call_webhook(workflow_id, payload)
            
            # Add task metadata
            result["task_id"] = task_id
            result["workflow_id"] = workflow_id
            result["timestamp"] = datetime.now().isoformat()
            
            # Record the task
            self.tasks_executed.append({
                "task": "call_n8n_workflow",
                "task_id": task_id,
                "status": "completed" if result.get("success", False) else "failed",
                "timestamp": datetime.now().isoformat(),
                "workflow_id": workflow_id,
                "success": result.get("success", False)
            })
            
            self.results[task_id] = result
            return result
            
        except Exception as e:
            error_result = {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "workflow_id": workflow_id
            }
            
            self.tasks_executed.append({
                "task": "call_n8n_workflow",
                "task_id": task_id,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "workflow_id": workflow_id
            })
            
            self.results[task_id] = error_result
            return error_result
    
    def send_notification(self, notification_type: str, message: str, 
                         **kwargs) -> Dict[str, Any]:
        """
        Send a notification via n8n.
        
        Args:
            notification_type: Type of notification (slack, email, etc.)
            message: The message to send
            **kwargs: Additional parameters for the notification
            
        Returns:
            Dictionary containing the notification result
        """
        task_id = f"notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            if notification_type.lower() == "slack":
                result = self.n8n_client.send_slack_notification(
                    message=message,
                    channel=kwargs.get("channel", "#alerts"),
                    severity=kwargs.get("severity", "info")
                )
            elif notification_type.lower() == "email":
                result = self.n8n_client.send_email_alert(
                    subject=kwargs.get("subject", "GSK QA Copilot Alert"),
                    body=message,
                    recipients=kwargs.get("recipients", []),
                    priority=kwargs.get("priority", "normal")
                )
            else:
                result = {
                    "success": False,
                    "error": f"Unsupported notification type: {notification_type}"
                }
            
            # Add task metadata
            result["task_id"] = task_id
            result["notification_type"] = notification_type
            result["timestamp"] = datetime.now().isoformat()
            
            # Record the task
            self.tasks_executed.append({
                "task": "send_notification",
                "task_id": task_id,
                "status": "completed" if result.get("success", False) else "failed",
                "timestamp": datetime.now().isoformat(),
                "notification_type": notification_type,
                "success": result.get("success", False)
            })
            
            self.results[task_id] = result
            return result
            
        except Exception as e:
            error_result = {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "notification_type": notification_type
            }
            
            self.tasks_executed.append({
                "task": "send_notification",
                "task_id": task_id,
                "status": "failed",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "notification_type": notification_type
            })
            
            self.results[task_id] = error_result
            return error_result
    
    def execute_plan_steps(self, plan_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Execute a list of planned steps.
        
        Args:
            plan_steps: List of step dictionaries with action and parameters
            
        Returns:
            Dictionary containing execution results
        """
        execution_id = f"plan_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution_results = []
        
        for i, step in enumerate(plan_steps):
            step_id = f"{execution_id}_step_{i+1}"
            action = step.get("action", "")
            parameters = step.get("parameters", {})
            
            try:
                if action == "fetch_logs":
                    result = self.fetch_logs(parameters.get("source", "data/synthetic_batches"))
                elif action == "run_deviation_detector":
                    result = self.run_deviation_detector(parameters.get("source", "data/synthetic_batches"))
                elif action == "call_n8n_workflow":
                    result = self.call_n8n_workflow(
                        parameters.get("workflow_id", ""),
                        parameters.get("payload", {})
                    )
                elif action == "send_notification":
                    result = self.send_notification(
                        parameters.get("type", "slack"),
                        parameters.get("message", ""),
                        **parameters.get("kwargs", {})
                    )
                else:
                    result = {
                        "task_id": step_id,
                        "status": "failed",
                        "error": f"Unknown action: {action}"
                    }
                
                execution_results.append({
                    "step_id": step_id,
                    "step_number": i + 1,
                    "action": action,
                    "result": result
                })
                
            except Exception as e:
                execution_results.append({
                    "step_id": step_id,
                    "step_number": i + 1,
                    "action": action,
                    "result": {
                        "task_id": step_id,
                        "status": "failed",
                        "error": str(e)
                    }
                })
        
        return {
            "execution_id": execution_id,
            "total_steps": len(plan_steps),
            "results": execution_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_tasks_executed(self) -> List[Dict[str, Any]]:
        """Get list of all tasks executed by this executor."""
        return self.tasks_executed.copy()
    
    def get_results(self) -> Dict[str, Any]:
        """Get all results from executed tasks."""
        return self.results.copy()
    
    def clear_history(self):
        """Clear task history and results."""
        self.tasks_executed = []
        self.results = {}
    
    def _group_deviations_by_reason(self, deviations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group deviations by reason for summary statistics."""
        grouped = {}
        for dev in deviations:
            reason = dev.get('deviation_reason', 'Unknown')
            grouped[reason] = grouped.get(reason, 0) + 1
        return grouped
    
    def _group_deviations_by_file(self, deviations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Group deviations by source file for summary statistics."""
        grouped = {}
        for dev in deviations:
            file_name = dev.get('source_file', 'Unknown')
            grouped[file_name] = grouped.get(file_name, 0) + 1
        return grouped


if __name__ == "__main__":
    # Test the executor
    executor = ExecutorAgent()
    
    # Test fetching logs
    print("Testing fetch_logs...")
    log_result = executor.fetch_logs("data/synthetic_batches")
    print(f"Log fetch result: {log_result['status']}")
    
    # Test deviation detection
    print("\nTesting deviation detection...")
    deviation_result = executor.run_deviation_detector("data/synthetic_batches")
    print(f"Deviation detection result: {deviation_result['status']}")
    print(f"Found {deviation_result.get('deviation_count', 0)} deviations")
    
    # Test n8n workflow call
    print("\nTesting n8n workflow call...")
    workflow_result = executor.call_n8n_workflow("test", {"test": True})
    print(f"N8N workflow result: {workflow_result.get('success', False)}")
    
    # Show executed tasks
    print(f"\nExecuted {len(executor.get_tasks_executed())} tasks:")
    for task in executor.get_tasks_executed():
        print(f"  - {task['task']}: {task['status']}")

