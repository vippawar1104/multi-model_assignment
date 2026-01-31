# 🚀 Streamlit App Quick Start Guide

## Two Streamlit Apps Available

### 1. **streamlit_app.py** (Recommended for Demo)
Clean, production-ready interface focused on Q&A functionality.

**Features:**
- ✅ Clean Q&A interface
- ✅ Real-time search & answer generation
- ✅ Source attribution with page numbers
- ✅ Performance metrics
- ✅ Chat history
- ✅ Export functionality

**Launch:**
```bash
streamlit run streamlit_app.py
```

### 2. **app_enhanced.py** (Full Featured)
Complete multi-modal interface with all features.

**Features:**
- ✅ Everything from streamlit_app.py PLUS:
- ✅ Multi-modal browser (images, tables, text)
- ✅ Evaluation metrics dashboard
- ✅ Analytics visualization
- ✅ Advanced chunk filtering
- ✅ Visual document preview

**Launch:**
```bash
streamlit run app_enhanced.py
```

## Quick Setup (2 Minutes)

### Step 1: Set API Key
You have three options:

**Option A: .env File (Recommended - Most Secure)**
1. Create a `.env` file in the project root (or copy from `.env.example`)
2. Add your API key:
   ```
   GROQ_API_KEY=your_actual_groq_api_key_here
   ```
3. Launch the app (it will automatically load from `.env`)
   ```bash
   streamlit run streamlit_app.py
   ```

**Option B: Environment Variable**
```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
streamlit run streamlit_app.py
```

**Option C: Enter in UI**
1. Launch the app
2. Enter API key in the sidebar
3. Start asking questions

### Step 2: Verify Data
Ensure you have processed chunks:
```bash
ls -lh data/processed/extracted_chunks.json
# Should show ~2.5MB file with 710 chunks
```

### Step 3: Launch & Test
```bash
streamlit run streamlit_app.py
```

Browser will open at `http://localhost:8501`

## Sample Questions to Try

### Economic Questions
- "What is Qatar's GDP growth forecast?"
- "What is the inflation rate in Qatar?"
- "Tell me about Qatar's fiscal policy"

### Specific Data Questions  
- "What are the hydrocarbon export figures?"
- "What monetary policy recommendations were made?"
- "What are the main economic risks?"

### Analytical Questions
- "Summarize the key findings of the IMF report"
- "What are the main challenges facing Qatar's economy?"

## Interface Overview

### Sidebar Configuration
- **API Key**: Enter your Groq or OpenAI API key
- **LLM Provider**: Choose between Groq (fast) or OpenAI
- **Top-K Chunks**: Control how many chunks to retrieve (3-15)
- **System Status**: Shows chunks loaded, query count
- **Document Info**: Quick stats about the processed document

### Main Interface
- **Sample Questions**: Click any sample to auto-fill
- **Query Input**: Type your question
- **Search & Answer**: Generates response with sources
- **Answer Display**: Shows LLM-generated answer
- **Metrics**: Latency, chunks used, sources
- **Sources**: Top 3 sources with page numbers and content preview
- **Chat History**: Recent 5 questions for reference

### Analytics (app_enhanced.py only)
- **Performance Metrics**: Response times, chunks usage
- **Quality Metrics**: Precision, relevance, accuracy
- **Multi-Modal Browser**: View images, tables, text chunks
- **Evaluation Dashboard**: Comprehensive metrics from Phase 8

## Performance Tips

### For Faster Responses
- Use **Groq** provider (1-2s latency)
- Set top-k to **5** (balance between quality and speed)
- Keep questions specific

### For Better Accuracy
- Increase top-k to **10-15**
- Use specific terms from the document
- Reference page numbers if known

### For Development/Testing
- Monitor the console for detailed logs
- Check data/processed/extracted_chunks.json if no results
- Verify API key is set correctly

## Troubleshooting

### "No chunks found"
```bash
# Check if chunks file exists
ls data/processed/extracted_chunks.json

# If missing, run processing pipeline
python simple_rag.py
```

### "API Error"
- Verify API key is correct
- Check internet connection
- Try switching provider (Groq ↔ OpenAI)

### "No relevant chunks"
- Question might be out of scope (document is about Qatar economics)
- Try rephrasing with different keywords
- Lower the specificity

### Port Already in Use
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

## Production Deployment

### Local Network Access
```bash
streamlit run streamlit_app.py --server.address 0.0.0.0
```

### Docker Deployment
```bash
docker-compose up
# App available at http://localhost:8501
```

### Cloud Deployment (Streamlit Cloud)
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add GROQ_API_KEY to secrets
4. Deploy!

## Key Features Demonstrated

### Assignment Requirements Checklist
- ✅ Multi-modal ingestion (text, tables, images)
- ✅ Vector indexing (FAISS)
- ✅ Smart chunking (semantic chunking)
- ✅ QA chatbot (interactive Q&A)
- ✅ Source attribution (page numbers, citations)
- ✅ Evaluation suite (metrics dashboard)

### Technical Highlights
- **Hybrid Retrieval**: Keyword search with weighted scoring
- **Stop Word Filtering**: Improves search relevance
- **Multi-level Scoring**: Exact phrase (10pt), word match (2pt), partial (1pt)
- **Source Attribution**: Every answer shows source pages
- **Performance Tracking**: Latency, throughput, quality metrics
- **Production Ready**: No hardcoded examples, real data only

## Video Demo Script

For the 3-5 minute assignment video:

### Part 1: Introduction (30s)
- Show Streamlit interface
- Highlight key features
- Explain multi-modal capabilities

### Part 2: Live Demo (2-3min)
- Ask 3-4 sample questions
- Show answer generation in real-time
- Demonstrate source attribution
- View multi-modal content (images, tables)

### Part 3: Technical Overview (1min)
- Show analytics dashboard
- Explain evaluation metrics
- Highlight performance stats

### Part 4: Conclusion (30s)
- Summarize capabilities
- Show all 8 phases complete
- Point to technical documentation

## Next Steps

### For Assignment Submission
1. ✅ Run `streamlit run app_enhanced.py`
2. ✅ Test with sample questions
3. ✅ Record video demonstration
4. ✅ Write technical report (2 pages)
5. ✅ Package codebase for submission

### For Further Development
- Add user authentication
- Implement feedback collection
- Add more LLM providers
- Create API endpoints
- Deploy to cloud

## Support

- **Documentation**: See README.md for full project details
- **Phase Details**: Check docs/ folder for phase-specific docs
- **Examples**: See examples/ folder for code samples
- **Testing**: Use simple_rag.py for CLI testing

---

**Ready to demo!** 🚀

All 8 phases complete, production-ready system with comprehensive Streamlit interface.
