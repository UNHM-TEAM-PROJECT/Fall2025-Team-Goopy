# UNH Graduate Catalog RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system designed to answer questions about the UNH Graduate Catalog using semantic search, contextual retrieval, and fine-tuned language models.

## Features

- **Semantic Search**: Uses sentence transformers for embedding-based retrieval
- **Tiered Retrieval System**: Prioritizes academic regulations, general info, and program-specific content
- **Query Enhancement**: Expands acronyms and boosts key terms for better retrieval
- **Re-ranking Pipeline**: Uses cross-encoder models and TF-IDF for result refinement
- **Synthetic Q&A Generation**: Automatically generates Q&A pairs from catalog content
- **Fine-tuned Model**: Custom-trained FLAN-T5 model for catalog-specific answers
- **Text Fragment URLs**: Direct links to specific sections in catalog pages
- **Automated Testing**: Gold standard evaluation with multiple metrics
- **Dashboard**: Real-time monitoring of test results and system performance

## Prerequisites

- Python 3.9 or higher
- Node.js 18+ (for frontend)
- 8GB+ RAM recommended
- Optional: NVIDIA GPU with CUDA for faster training

## Quick Start

### 1. Backend Setup

```bash
# Clone the repository
git clone <repository-url>
cd UNH_chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
cd backend
pip install -r requirements.txt

# Initialize configuration
python -c "from config.settings import load_retrieval_config; load_retrieval_config()"

# Start the server
python main.py
```

The backend will start on `http://localhost:8003`

### 2. Frontend Setup

```bash
# In a new terminal
cd frontend

# Install dependencies
npm install

# Build the frontend
npm run build

# The built files will be served by the FastAPI backend
```

### 3. Access the Application

- **Chat Interface**: http://localhost:8003
- **Dashboard**: http://localhost:8003/dashboard
- **API Docs**: http://localhost:8003/docs

**Run Containerized (Optional)**
```bash
docker system prune -a --volumes # Free space before building (optional)
docker build -t goopy-app .
docker run -p 8003:8003 --name goopy-app -e PUBLIC_URL=http://localhost:8003/ goopy-app
```

### Deploy on Server

```bash
git clone https://github.com/UNHM-TEAM-PROJECT/Fall2025-Team-Goopy.git
cd Fall2025-Team-Goopy
docker system prune -a --volumes
docker build -t goopy-app .
docker run -d \
  --name goopy-app \
  -p 8003:8003 \
  -v $(pwd)/backend/train/models:/app/backend/train/models \
  goopy-app
```

## Architecture

### Backend Structure

```
backend/
├── config/                 # Configuration files
│   ├── retrieval.yaml     # Main retrieval settings
│   ├── query_rewrite.json # Query transformation rules
│   └── settings.py        # Configuration loader
├── models/                 # ML model management
│   ├── api_models.py      # Pydantic request/response models
│   └── ml_models.py       # Model initialization
├── services/               # Core business logic
│   ├── chunk_service.py   # Document chunking and indexing
│   ├── retrieval_service.py # Semantic search
│   ├── qa_service.py      # Answer generation
│   ├── query_pipeline.py  # End-to-end query processing
│   ├── reranking_service.py # Result re-ranking
│   ├── compression_service.py # Context compression
│   ├── synthetic_qa_service.py # Q&A generation
│   ├── gold_set_service.py # Evaluation dataset
│   └── calendar_fallback.py # Date/deadline handling
├── routers/               # API endpoints
│   ├── chat.py           # Chat endpoints
│   └── dashboard.py      # Dashboard endpoints
├── train/                # Model training
│   ├── train.py         # Training script
│   └── data/            # Training data
└── main.py              # Application entry point
```

### Key Components

**Retrieval Pipeline:**
1. Query Enhancement → Expand acronyms, boost terms
2. Semantic Search → Vector similarity search
3. Tier Boosting → Prioritize academic regulations
4. Re-ranking → Cross-encoder + TF-IDF refinement
5. Context Compression → Extract relevant sentences
6. Answer Generation → Fine-tuned FLAN-T5 model

**Tiered Content Hierarchy:**
- **Tier 0**: Gold standard Q&A pairs (highest priority)
- **Tier 1**: Academic regulations and policies
- **Tier 2**: General graduate information
- **Tier 3**: Course descriptions
- **Tier 4**: Program-specific content

## Configuration

Edit `backend/config/retrieval.yaml` to customize:

```yaml
# Retrieval settings
retrieval_sizes:
  topn_default: 120  # Candidates to retrieve
  k: 5               # Final results to return

# Tier boost multipliers
tier_boosts:
  0: 3.0   # Gold set
  1: 1.5   # Academic regulations
  2: 1.2   # General info
  3: 1.0   # Course descriptions
  4: 1.0   # Program-specific

# Enable/disable features
enhancements:
  enabled: false  # Master switch for query enhancement
  query_enhancement:
    enabled: false
  reranking:
    enabled: false
  compression:
    enabled: false

# Answer generation
performance:
  max_tokens: 200
  use_finetuned_model: false  # Use fine-tuned or base model
```

## Training Custom Models

### Fine-tune the Answer Model

```bash
cd backend/train

# Prepare training data in train/data/*.json
# Format: [{"query": "...", "answer": "...", "url": "..."}]

# Train with default seed
python train.py

# Train with custom seed for reproducibility
python train.py --seed 123
```

The training script will:
- Generate training examples using retrieval pipeline
- Split into train/validation sets (80/20)
- Fine-tune FLAN-T5-small model
- Save to `backend/train/models/flan-t5-small-finetuned/`
- Use GPU if available, fallback to CPU

**Training Output:**
- Model checkpoints in `train/models/`
- Verification files (`*.out`) showing generated answers
- Training logs and evaluation metrics

### Generate Synthetic Q&A Pairs

Enable in `retrieval.yaml`:

```yaml
synthetic_qa:
  enabled: true
  boost_synthetic_qa: 1.3
  generate_for_tiers: [1, 2]  # Only important chunks
```

This generates question-answer pairs from catalog content to improve semantic matching.

## Automated Testing

### Run Evaluation Tests

```bash
cd automation_testing
python run_tests.py
```

**Metrics Computed:**
- **Nugget-based**: Precision, Recall, F1 (key information coverage)
- **Semantic**: SBERT cosine similarity (meaning preservation)
- **Token-level**: BERTScore F1 (lexical quality)
- **Retrieval**: Recall@k, NDCG@k (ranking quality)

**View Results:**
- Dashboard: http://localhost:8003/dashboard
- Reports: `automation_testing/reports/TIMESTAMP/`
- Raw predictions: `automation_testing/reports/TIMESTAMP/preds.jsonl`

### Gold Standard Format

Add test cases to `automation_testing/gold.jsonl`:

```json
{
  "id": "category:q1",
  "query": "What is the minimum GPA for graduate students?",
  "reference_answer": "Graduate students must maintain a 3.0 GPA.",
  "nuggets": ["3.0 GPA", "graduate students", "maintain"],
  "gold_passages": ["chunk_id_1", "chunk_id_2"],
  "url": "https://catalog.unh.edu/graduate/..."
}
```

## Dashboard Features

The dashboard provides:
- **Test Results**: View all test runs with summary statistics
- **Per-Question Analysis**: Detailed metrics for each question
- **Category Breakdown**: Performance by question category
- **Predictions**: Compare model answers to reference answers
- **Retrieval Analysis**: Which chunks were retrieved for each question

## API Endpoints

### Chat Endpoint
```bash
POST /chat
Headers: X-Session-Id: <session-id>
Body: {
  "message": "What is the minimum GPA?",
  "history": []
}

Response: {
  "answer": "Graduate students must maintain a 3.0 GPA.",
  "sources": ["https://catalog.unh.edu/..."],
  "retrieval_path": [...],
  "transformed_query": null
}
```

### Dashboard Endpoints
```bash
GET  /reports              # Get all test results
POST /run-tests            # Trigger new test run
GET  /dashboard            # Serve dashboard HTML
POST /reset-session        # Clear session
POST /reset                # Clear all sessions
```

## Troubleshooting

### Common Issues

**Cache Issues**
- Delete `scraper/chunks_cache.pkl` to rebuild the index
- The cache stores embeddings and chunk data

**Memory Issues**
- Reduce `topn_default` in `retrieval.yaml`
- Disable synthetic Q&A generation
- Use smaller batch sizes in training

**Slow Performance**
- Enable caching in `retrieval.yaml`
- Reduce `max_tokens` for faster generation
- Use GPU for training/inference

**Model Not Found**
- Run training script to create fine-tuned model
- Or set `use_finetuned_model: false` to use base model

**Container Closing Randomly**
- run monitor.sh to generate scripts to help maintain container uptime with real time alerts to email or Teams chats

## Development

### Adding New Features

1. **New Retrieval Strategy**: Modify `services/retrieval_service.py`
2. **Custom Reranking**: Update `services/reranking_service.py`
3. **Query Preprocessing**: Edit `services/query_enhancement.py`
4. **Answer Post-processing**: Modify `services/qa_service.py`

### Running Tests

```bash
# Backend tests
cd backend
python -m pytest tests/

# Evaluation tests
cd automation_testing
python run_tests.py
```

### Code Style

```bash
# Format code
black backend/
isort backend/

# Lint
flake8 backend/
pylint backend/
```

## Dependencies

**Core:**
- FastAPI, Uvicorn (web framework)
- PyTorch (deep learning)
- Transformers (language models)
- Sentence-Transformers (embeddings)
- Scikit-learn (ML utilities)
- LangChain (document processing)

**Models Used:**
- `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- `cross-encoder/ms-marco-MiniLM-L-6-v2` (reranking)
- `google/flan-t5-small` (answer generation)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is developed for the UNH Graduate School.

## Acknowledgments

- UNH Graduate School for catalog data
- Hugging Face for transformer models
- Sentence-Transformers library
- FastAPI framework

## Contact

For questions or issues, please open an issue on GitHub or contact the UNH Graduate School IT team.
