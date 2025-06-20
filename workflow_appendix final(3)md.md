# Appendix: NLP and Agentic AI Workflow for Electrophysiology

## Figure A1: Complete Workflow for NLP and Agentic AI in Electrophysiology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      DATA ACQUISITION & PREPROCESSING                   │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐ │
│ │ Literature Sources │  │ Clinical Guidelines│  │ Electronic Health    │ │
│ │ - PubMed          │  │ - ACC/AHA          │  │ Records              │ │
│ │ - Cochrane        │  │ - ESC              │  │ - EP Study Reports   │ │
│ │ - EP Journals     │  │ - HRS              │  │ - Patient Notes      │ │
│ └────────┬──────────┘  └────────┬──────────┘  └──────────┬────────────┘ │
│          │                      │                        │              │
│          ▼                      ▼                        ▼              │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │                     Automated Data Collection                      │   │
│ │ - Web Scraping (BeautifulSoup, Selenium)                          │   │
│ │ - API Access (PubMed API, EHR APIs)                               │   │
│ │ - PDF Text Extraction (PyPDF2, pdfminer)                          │   │
│ └────────────────────────────────┬──────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │                     Text Preprocessing Pipeline                    │   │
│ │ - Tokenization                                                     │   │
│ │ - Sentence Segmentation                                            │   │
│ │ - Noise Removal                                                    │   │
│ │ - Domain-Specific Normalization (e.g., medical abbreviations)      │   │
│ └────────────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       NLP PROCESSING & ANALYSIS                         │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐ │
│ │ Named Entity      │  │ Relation          │  │ Text                  │ │
│ │ Recognition (NER) │  │ Extraction        │  │ Summarization         │ │
│ │                   │  │                   │  │                       │ │
│ │ - BioBERT-EP      │  │ - Dependency      │  │ - Extractive         │ │
│ │ - EP-specific     │  │   Parsing         │  │   Summarization      │ │
│ │   entities:       │  │ - Co-occurrence   │  │ - Abstractive        │ │
│ │   • Arrhythmias   │  │   Analysis        │  │   Summarization      │ │
│ │   • Medications   │  │ - Causal          │  │ - Key Points         │ │
│ │   • Procedures    │  │   Relationships   │  │   Extraction         │ │
│ │   • Measurements  │  │                   │  │                       │ │
│ └────────┬──────────┘  └────────┬──────────┘  └──────────┬────────────┘ │
│          │                      │                        │              │
│          │                      │                        │              │
│ ┌────────▼──────────┐  ┌────────▼──────────┐  ┌──────────▼────────────┐ │
│ │ Sentiment         │  │ Classification    │  │ Question              │ │
│ │ Analysis          │  │ & Categorization  │  │ Answering             │ │
│ │                   │  │                   │  │                       │ │
│ │ - Opinion Mining  │  │ - Evidence        │  │ - Clinical Query      │ │
│ │ - Treatment       │  │   Grading         │  │   Processing          │ │
│ │   Perception      │  │ - Recommendation  │  │ - Evidence-Based      │ │
│ │ - Research Trend  │  │   Strength        │  │   Responses           │ │
│ │   Analysis        │  │ - Topic Modeling  │  │                       │ │
│ └────────┬──────────┘  └────────┬──────────┘  └──────────┬────────────┘ │
└──────────┼───────────────────────┼───────────────────────┼──────────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       AGENTIC AI INTEGRATION                            │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │                     Knowledge Base Construction                    │   │
│ │ - Structured Database of Extracted Information                     │   │
│ │ - Ontology Mapping (SNOMED CT, MeSH)                              │   │
│ │ - Knowledge Graph Construction                                     │   │
│ └────────────────────────────────┬──────────────────────────────────┘   │
│                                  │                                      │
│                                  ▼                                      │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │                     Autonomous Agent System                        │   │
│ │                                                                    │   │
│ │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │   │
│ │  │ Literature      │    │ Guideline       │    │ Reporting       │ │   │
│ │  │ Agent           │    │ Agent           │    │ Agent           │ │   │
│ │  │                 │    │                 │    │                 │ │   │
│ │  │ - Monitors new  │    │ - Tracks        │    │ - Generates     │ │   │
│ │  │   publications  │    │   guideline     │    │   summaries     │ │   │
│ │  │ - Extracts key  │    │   updates       │    │ - Creates       │ │   │
│ │  │   findings      │    │ - Compares      │    │   evidence      │ │   │
│ │  │ - Alerts on     │    │   recommendations│   │   tables        │ │   │
│ │  │   breakthroughs │    │   across sources│    │ - Formats for   │ │   │
│ │  │                 │    │                 │    │   sharing       │ │   │
│ │  └─────────────────┘    └─────────────────┘    └─────────────────┘ │   │
│ │                                                                    │   │
│ │  ┌─────────────────────────────────────────────────────────────┐   │   │
│ │  │                  Coordination Layer                         │   │   │
│ │  │ - Task Prioritization                                       │   │   │
│ │  │ - Resource Allocation                                       │   │   │
│ │  │ - Exception Handling                                        │   │   │
│ │  │ - User Preference Learning                                  │   │   │
│ │  └─────────────────────────────────────────────────────────────┘   │   │
│ └────────────────────────────────┬──────────────────────────────────┘   │
└─────────────────────────────────┬─────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       CLINICAL APPLICATIONS                             │
├─────────────────────────────────────────────────────────────────────────┤
│ ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────────┐ │
│ │ Research Support  │  │ Clinical Decision │  │ Education &           │ │
│ │                   │  │ Support           │  │ Training              │ │
│ │ - Literature      │  │                   │  │                       │ │
│ │   Reviews         │  │ - Treatment       │  │ - Personalized        │ │
│ │ - Meta-analyses   │  │   Recommendations │  │   Learning            │ │
│ │ - Grant Writing   │  │ - Risk            │  │ - Case Studies        │ │
│ │   Support         │  │   Stratification  │  │ - Simulation          │ │
│ │                   │  │ - Protocol        │  │   Training            │ │
│ │                   │  │   Adherence       │  │                       │ │
│ └───────────────────┘  └───────────────────┘  └───────────────────────┘ │
│                                                                         │
│ ┌───────────────────────────────────────────────────────────────────┐   │
│ │                     User Interfaces & Delivery                     │   │
│ │ - Web Dashboards                                                   │   │
│ │ - Mobile Applications                                              │   │
│ │ - EHR Integration                                                  │   │
│ │ - Email/Slack Notifications                                        │   │
│ │ - PDF/CSV Exports                                                  │   │
│ └───────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Figure A2: Sample Code Implementation for EP-NER

```python
# Named Entity Recognition for Electrophysiology Text
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Load BioBERT model fine-tuned for EP entity recognition
tokenizer = AutoTokenizer.from_pretrained("ep-research/biobert-ep-ner")
model = AutoModelForTokenClassification.from_pretrained("ep-research/biobert-ep-ner")

# Create NER pipeline
ep_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Sample EP text from clinical study
ep_text = """
Purpose: Sevoflurane is known to prolong the QT interval. 
This study aimed to determine the effect of the interaction 
between intravenous anesthetics and sevoflurane on the QT interval.

Methods: The study included 48 patients who underwent lumbar spine surgery. 
Patients received 3 μg/kg fentanyl and were then randomly allocated to 
either Group T, in which they received 5 mg/kg thiamylal, or Group P, 
in which they received 1.5 mg/kg propofol, at 2 min after administration 
of fentanyl injection for anesthetic induction.
"""

# Extract entities
entities = ep_ner(ep_text)

# Process and display results
ep_entities = {}
for entity in entities:
    category = entity["entity_group"]
    value = entity["word"]
    if category not in ep_entities:
        ep_entities[category] = []
    ep_entities[category].append(value)

print("Extracted EP Entities:")
for category, values in ep_entities.items():
    print(f"{category}: {', '.join(values)}")

# Output would include:
# MEDICATION: Sevoflurane, fentanyl, thiamylal, propofol
# MEASUREMENT: QT interval
# PROCEDURE: anesthetic induction, lumbar spine surgery
# DOSAGE: 3 μg/kg, 5 mg/kg, 1.5 mg/kg
# PATIENT_GROUP: Group T, Group P
```

## Figure A3: Agentic AI Implementation for EP Literature Monitoring

```python
# EP Literature Agent Implementation
import os
import schedule
import time
import requests
import pandas as pd
from datetime import datetime
from transformers import pipeline

class EPLiteratureAgent:
    def __init__(self):
        # Initialize NLP components
        self.summarizer = pipeline("summarization", model="ep-research/biobert-ep-summarizer")
        self.ner = pipeline("ner", model="ep-research/biobert-ep-ner")
        
        # Knowledge base
        self.db_path = "ep_literature_db.csv"
        if not os.path.exists(self.db_path):
            pd.DataFrame(columns=["pmid", "title", "abstract", "date", "entities", 
                                 "summary", "relevance_score"]).to_csv(self.db_path, index=False)
        
        # User preferences (would be loaded from user profile)
        self.topics_of_interest = ["atrial fibrillation", "catheter ablation", 
                                  "QT interval", "ventricular tachycardia"]
        self.min_relevance_score = 0.7
        
    def search_new_publications(self):
        """Query PubMed API for new EP publications"""
        print(f"[{datetime.now()}] Searching for new publications...")
        
        # In production, would use proper PubMed API
        # This is a simplified example
        query = " OR ".join([f'"{topic}"[Title/Abstract]' for topic in self.topics_of_interest])
        query += ' AND ("last 7 days"[PDat])'
        
        # Simulate API response with sample data
        sample_results = [
            {
                "pmid": "36123456",
                "title": "Novel Mapping Techniques for Ventricular Tachycardia Ablation",
                "abstract": "This study evaluates a new approach to electroanatomic mapping...",
                "date": "2025-05-15"
            },
            {
                "pmid": "36123457",
                "title": "Long-term Outcomes of Catheter Ablation for Atrial Fibrillation",
                "abstract": "In this 5-year follow-up study of 500 patients who underwent...",
                "date": "2025-05-14"
            }
        ]
        
        return sample_results
    
    def process_publication(self, publication):
        """Process a single publication with NLP pipeline"""
        # Extract entities
        text = publication["title"] + " " + publication["abstract"]
        entities = self.ner(text)
        
        # Generate summary
        summary = self.summarizer(text, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        
        # Calculate relevance score based on entity matching with topics of interest
        relevance_score = self.calculate_relevance(entities, publication["title"])
        
        # Add to database
        publication["entities"] = entities
        publication["summary"] = summary
        publication["relevance_score"] = relevance_score
        
        return publication
    
    def calculate_relevance(self, entities, title):
        """Calculate relevance score based on entities and user preferences"""
        # Simplified scoring algorithm
        score = 0.0
        
        # Check for topic matches in title
        for topic in self.topics_of_interest:
            if topic.lower() in title.lower():
                score += 0.4
                break
                
        # Check for relevant entities
        relevant_entity_count = 0
        for entity in entities:
            if entity["entity_group"] in ["ARRHYTHMIA", "PROCEDURE", "MEDICATION"]:
                relevant_entity_count += 1
        
        # Add entity-based score component
        score += min(0.6, relevant_entity_count * 0.1)
        
        return min(1.0, score)
    
    def generate_alerts(self, new_publications):
        """Generate alerts for highly relevant publications"""
        alerts = []
        for pub in new_publications:
            if pub["relevance_score"] >= self.min_relevance_score:
                alerts.append({
                    "title": pub["title"],
                    "summary": pub["summary"],
                    "relevance": pub["relevance_score"],
                    "pmid": pub["pmid"]
                })
        return alerts
    
    def run_daily_update(self):
        """Main workflow that runs on schedule"""
        # Search for new publications
        new_pubs = self.search_new_publications()
        
        # Process each publication
        processed_pubs = []
        for pub in new_pubs:
            processed_pub = self.process_publication(pub)
            processed_pubs.append(processed_pub)
        
        # Update database
        current_db = pd.read_csv(self.db_path)
        new_records = pd.DataFrame(processed_pubs)
        updated_db = pd.concat([current_db, new_records]).drop_duplicates(subset=["pmid"])
        updated_db.to_csv(self.db_path, index=False)
        
 
(Content truncated due to size limit. Use line ranges to read in chunks)