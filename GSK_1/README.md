# GSK_1: Multi-Agent QA System for Pharma Manufacturing

A sophisticated Quality Assurance (QA) prototype system for pharmaceutical manufacturing that leverages multi-agent orchestration with LangGraph and Groq LLM to analyze batch data, retrieve SOPs (Standard Operating Procedures), and detect deviations in manufacturing processes.

## 🎯 Project Overview

GSK_1 is an intelligent QA system designed to streamline pharmaceutical quality assurance by:
- **Planning** execution strategies based on user queries
- **Retrieving** relevant SOP/GMP documents and batch data
- **Analyzing** manufacturing deviations and anomalies
- **Consolidating** insights for decision-making

The system uses a multi-agent architecture where specialized agents handle different aspects of the QA workflow, coordinated through LangGraph.

## 📁 Project Structure

```
GSK_1/
├── backend/                           # Core backend system
│   ├── main.py                       # Multi-agent orchestrator using LangGraph
│   ├── agents/                       # Specialized agents
│   │   ├── planner.py               # Query planning agent (Groq LLM)
│   │   ├── retriever.py             # Document & data retrieval agent
│   │   └── executor.py              # Query execution & response agent
│   └── services/                     # Business logic services
│       ├── SOP_GMP_retriver.py      # SOP/GMP document retrieval service
│       ├── deviation_detector.py    # Anomaly & deviation detection
│       └── index_documents_cohere.py # Document indexing with Cohere embeddings
├── data/                             # Data management
│   ├── synthetic_batches/           # Generated batch data (CSV format)
│   └── sops/                        # Standard Operating Procedures documentation
├── generate_synthetic.py             # Synthetic batch data generator
└── README.md                         # This file
```

## 🚀 Key Components

### 1. **Multi-Agent Orchestrator** (`main.py`)
- Orchestrates the workflow using LangGraph
- Manages state through the workflow pipeline
- Implements three-stage workflow:
  1. **Planner** → Creates execution plan
  2. **Retriever** → Executes data retrieval
  3. **Consolidator** → Prepares final output

### 2. **Agents** (`agents/`)

#### Planner Agent (`planner.py`)
- Uses Groq LLM to analyze user queries
- Creates structured execution plans
- Determines required agents and actions
- Supports complexity assessment

#### Retriever Agent (`retriever.py`)
- Retrieves SOP and GMP documents
- Analyzes batch manufacturing data
- Integrates with document indexing service
- Returns structured retrieval results

#### Executor Agent (`executor.py`)
- Executes planned queries
- Generates final responses
- Provides formatting and interpretation

### 3. **Services** (`services/`)

#### SOP/GMP Retriever Service
- Manages SOP and GMP document retrieval
- Supports semantic search and filtering
- Integrates with document embeddings

#### Deviation Detector Service
- Identifies manufacturing anomalies
- Analyzes batch data patterns
- Flags deviations from normal ranges

#### Document Indexing Service
- Indexes documents using Cohere embeddings
- Enables semantic search capabilities
- Manages document metadata

### 4. **Data Generation** (`generate_synthetic.py`)
Generates realistic pharmaceutical manufacturing data with various anomaly types:
- **SUDDEN_SPIKE**: Sharp temperature/pressure deviations
- **GRADUAL_DRIFT**: Slow metric degradation over time
- **SENSOR_FAILURE**: Missing sensor readings
- **CALIBRATION_DUE**: Equipment calibration alerts
- **SILENT_SPIKE**: Anomalies without logged notes

**Output**: CSV files in `data/synthetic_batches/` containing:
- Timestamp, batch ID, manufacturing step
- Equipment ID and operator information
- Metric values (temperature, pressure, pH, weight)
- Units and operational notes

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Required API keys: Groq, Cohere
- CSV batch data files

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/awinpavan/QA_Prototype.git
   cd QA_Prototype/GSK_1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies:
   - `langgraph` - Workflow orchestration
   - `langchain` - LLM framework
   - `langchain-groq` - Groq LLM integration
   - `cohere` - Embedding models
   - `python-dotenv` - Environment management

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Set the following in `.env`:
   ```
   GROQ_API_KEY=your_groq_api_key
   COHERE_API_KEY=your_cohere_api_key
   ```

4. **Generate synthetic data (optional)**
   ```bash
   cd GSK_1
   python generate_synthetic.py
   ```
   This creates sample batch data in `data/synthetic_batches/`

## 📖 Usage

### Interactive Query Mode

Run the main orchestrator:
```bash
python GSK_1/backend/main.py
```

Example queries:
- "What are the quality standards for batch processing?"
- "Analyze the recent batch data for anomalies"
- "What are the SOP guidelines for temperature monitoring?"

### Programmatic Usage

```python
from GSK_1.backend.main import MultiAgentOrchestrator

# Initialize orchestrator
orchestrator = MultiAgentOrchestrator()

# Process a query
results = orchestrator.process_query(
    "Analyze batch data for deviation from SOP guidelines"
)

# Access results
print(results["final_response"])
print(results["status"])
```

## 📊 Workflow Execution Flow

```
User Query
    ↓
[PLANNER AGENT] → Create Execution Plan
    ↓
[RETRIEVER AGENT] → Fetch SOP/GMP/Batch Data
    ↓
[CONSOLIDATOR] → Structure & Prepare Output
    ↓
[EXECUTOR AGENT] → Generate Response
    ↓
Final Response
```

## 🔍 Data Format

### Batch Data (CSV)
```csv
timestamp,batch_id,step,equipment_id,metric_name,metric_value,unit,operator,note
2024-07-01T10:30:00,batch_abc123,step_0,reactor-01,temperature,38.5,C,op_101,
2024-07-01T10:30:00,batch_abc123,step_0,reactor-01,pressure,145.2,PSI,op_101,Pressure rising unexpectedly
```

### Supported Metrics
- **Temperature**: 37-39°C (normal range)
- **Pressure**: 140-150 PSI (normal range)
- **pH**: 6.8-7.2 (normal range)
- **Scale Weight**: 500-502 kg (normal range)

## 🚨 Anomaly Detection

The system detects and flags five types of deviations:

| Anomaly Type | Description | Detection Method |
|---|---|---|
| **SUDDEN_SPIKE** | Abrupt metric deviation | Threshold comparison |
| **GRADUAL_DRIFT** | Slow metric degradation | Time-series analysis |
| **SENSOR_FAILURE** | Missing readings (60% probability) | Null value detection |
| **CALIBRATION_DUE** | Equipment needs calibration | SOP compliance check |
| **SILENT_SPIKE** | Anomaly without logged note | Pattern recognition |

## 📝 Environment Variables

```
# Groq LLM Configuration
GROQ_API_KEY=your_api_key
GROQ_MODEL=mixtral-8x7b-32768

# Cohere Embedding Configuration
COHERE_API_KEY=your_api_key
COHERE_MODEL=embed-english-v3.0

# System Configuration
DATA_PATH=./data
SOP_PATH=./data/sops
BATCH_DATA_PATH=./data/synthetic_batches
```

## 🧪 Testing & Validation

### Generate Test Data
```bash
python GSK_1/generate_synthetic.py
```

### Test Batch Types
- `batch_normal_01.csv` - Normal operation baseline
- `batch_anomaly_sudden_spike_01.csv` - Temperature spike anomaly
- `batch_anomaly_gradual_drift_01.csv` - Gradual temperature drift
- `batch_anomaly_sensor_failure_01.csv` - Sensor reading failures
- `batch_anomaly_calibration_due_01.csv` - Calibration due alert
- `batch_anomaly_silent_spike_01.csv` - Unlogged anomaly

## 🔧 Configuration Files

### Key Configuration Sections

**Metrics Configuration** (in `generate_synthetic.py`):
- Define normal ranges for each metric
- Set anomaly shift magnitudes
- Configure equipment IDs and operators

**Anomaly Windows**:
- Anomalies typically occur in steps 5-8 (configurable)
- Enables consistent test pattern generation

## 📚 API Reference

### MultiAgentOrchestrator

```python
orchestrator = MultiAgentOrchestrator()

# Main method
results = orchestrator.process_query(user_query: str) -> Dict[str, Any]

# Format output for display
output = orchestrator.format_output(results: Dict[str, Any]) -> str
```

### Return Structure
```python
{
    "status": "completed|error",
    "user_query": str,
    "execution_plan": ExecutionPlan,
    "plan_summary": Dict,
    "retrieval_results": List[Dict],
    "consolidated_data": Dict,
    "final_response": str,
    "errors": List[str],
    "completed_steps": List[int]
}
```

## 🐛 Troubleshooting

| Issue | Solution |
|---|---|
| API Key errors | Verify `GROQ_API_KEY` and `COHERE_API_KEY` in `.env` |
| Import errors | Run `pip install -r requirements.txt` |
| No data found | Generate synthetic data with `python generate_synthetic.py` |
| Agent initialization fails | Check LangGraph and LangChain versions compatibility |

## 🚀 Future Enhancements

- [ ] Real-time monitoring dashboard
- [ ] Integration with manufacturing execution systems (MES)
- [ ] Predictive deviation forecasting
- [ ] Advanced visualization of anomalies
- [ ] Database integration for batch history
- [ ] API endpoint exposure for external systems
- [ ] Custom anomaly threshold configuration UI
