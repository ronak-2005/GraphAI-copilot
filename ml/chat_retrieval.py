"""
CHAT RETRIEVAL LAYER
====================

PURPOSE:
    Retrieve pre-computed insights in response to user questions.
    This is RETRIEVAL ONLY - no computation, no analysis, no interpretation.

SCOPE:
    - Question normalization (lowercase, punctuation stripping)
    - Keyword-based insight matching
    - Heuristic graph association (fallback when upstream linking absent)
    - Evidence packaging with confidence scoring

EXPLICIT NON-SCOPE:
    - No data processing (no CSV loading, no statistics computation)
    - No graph interpretation (graphs are metadata references only)
    - No new insight generation (only returns pre-computed knowledge)
    - No autonomous reasoning (no logic chaining, no inference)

CURRENT LIMITATIONS:
    1. Graph-Insight Linking: Currently heuristic (keyword-based)
       - Depends on upstream 'supports_graphs' from insight_extraction.py
       - Fallback heuristic is transparent (marked as 'linking_method: heuristic')
       - Fix belongs in insight_extraction.py, not here
    
    2. Session Context: Passive history logging only
       - Stored but not used for disambiguation or follow-up resolution
       - Future session-aware retrieval belongs in higher orchestration layer
       - Do not add conversational intelligence here
    
    3. Confidence Semantics: Retrieval confidence, not statistical confidence
       - "High confidence" = strong keyword match
       - Does NOT imply correctness, completeness, or statistical certainty
       - UI must clarify this to users

TEMPORARY SCAFFOLDING (Will Be Removed):
    - interactive_mode(): Testing harness, will move to API/frontend
    - show_help(), show_context(): Demo utilities
    - save_session(): Should live in UI layer
    - batch_retrieve(), RetrievalValidator: Should move to /tests

CORRECT USAGE:
    retrieval = ChatRetrieval()
    retrieval.setup()
    response = retrieval.retrieve("How many rows?")
    
    # response contains:
    # - matched_insights: List of pre-computed insights with grounding
    # - related_graphs: List of graph references (heuristic linking)
    # - retrieval_confidence: Match strength (high/medium/low/none)
    # - metadata: Retrieval method and clarifying notes
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

class Config:
    INSIGHTS_FILE = "../ml/graph_insights/master_insights.json"
    SEARCH_INDEX_FILE = "../ml/graph_insights/search_index.json"

    TOP_K_INSIGHTS = 5
    TOP_K_GRAPHS = 3
    
    CONFIDENCE_THRESHOLD_HIGH = 0.6
    CONFIDENCE_THRESHOLD_MEDIUM = 0.3
    
    SESSION_FILE = "../ml/graph_insights/session_context.json"

class ChatRetrieval:
    
    def __init__(self):
        self.insights_data = None
        self.search_index = None
        self.session_context = []
        
    def setup(self):
        print("="*80)
        print("CHAT RETRIEVAL LAYER")
        print("="*80)
        print("Purpose: Retrieve pre-computed insights only")
        print("No computation | No analysis | No interpretation")
        print("="*80)
        
        self.load_insights()
        self.load_search_index()
        
        print("\n" + "="*80)
        print("READY")
        print("="*80)
    
    def load_insights(self):
        print("\nLoading Insights")
        
        insights_path = Path(Config.INSIGHTS_FILE)
        if not insights_path.exists():
            raise FileNotFoundError(f"Not found: {Config.INSIGHTS_FILE}")
        
        with open(insights_path, 'r') as f:
            self.insights_data = json.load(f)
        
        total_insights = len(self.insights_data.get('insights', []))
        total_graphs = len(self.insights_data.get('graph_catalog', []))
        
        print(f"   Insights: {total_insights}")
        print(f"   Graphs: {total_graphs}")
    
    def load_search_index(self):
        print("\nLoading Search Index")
        
        index_path = Path(Config.SEARCH_INDEX_FILE)
        if not index_path.exists():
            raise FileNotFoundError(f"Not found: {Config.SEARCH_INDEX_FILE}")
        
        with open(index_path, 'r') as f:
            self.search_index = json.load(f)
        
        print(f"   Entries: {len(self.search_index.get('searchable_content', []))}")
    
    def normalize_question(self, question):
        question = question.lower().strip()
        question = re.sub(r'[^\w\s]', ' ', question)
        return question
    
    def retrieve(self, question):
        print(f"\nQuestion: {question}")
        
        normalized_question = self.normalize_question(question)
        
        insights = self.keyword_search(normalized_question)
        
        graphs = self.find_related_graphs(insights, normalized_question)
        
        retrieval_confidence = self.calculate_confidence(insights)
        
        response = {
            "status": "success",
            "question": question,
            "matched_insights": insights[:Config.TOP_K_INSIGHTS],
            "related_graphs": graphs[:Config.TOP_K_GRAPHS],
            "retrieval_confidence": retrieval_confidence,
            "metadata": {
                "retrieval_method": "keyword_match",
                "total_matches": len(insights),
                "note": "Confidence reflects question-to-insight match strength, not data certainty"
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.session_context.append({
            "question": question,
            "response": response
        })
        
        return response
    
    def keyword_search(self, question):
        question_words = set(question.lower().split())
        
        results = []
        
        for item in self.search_index.get('searchable_content', []):
            if item['type'] != 'insight':
                continue
            
            keywords = set(item.get('keywords', []))
            content_words = set(item.get('content', '').lower().split())
            
            keyword_matches = len(question_words.intersection(keywords))
            content_matches = len(question_words.intersection(content_words))
            
            score = keyword_matches * 2 + content_matches
            
            if score > 0:
                full_insight = self.get_full_insight(item['id'])
                
                if full_insight:
                    insight_detail = {
                        "id": item.get('id'),
                        "insight": full_insight.get('insight', item['content']),
                        "category": full_insight.get('category', item['category']),
                        "importance": full_insight.get('importance', 'medium'),
                        "task_type": full_insight.get('task_type'),
                        "grounding": full_insight.get('grounding', {}),
                        "supports_graphs": full_insight.get('supports_graphs', []),
                        "match_strength": min(score / 10, 1.0),
                        "raw_score": score
                    }
                    
                    results.append(insight_detail)
        
        results.sort(key=lambda x: x['match_strength'], reverse=True)
        
        return results
    
    def get_full_insight(self, insight_id):
        for insight in self.insights_data.get('insights', []):
            if insight.get('id') == insight_id:
                return insight
        return None
    
    def find_related_graphs(self, insights, question):
        graph_scores = Counter()
        
        for insight in insights[:Config.TOP_K_INSIGHTS]:
            for graph_id in insight.get('supports_graphs', []):
                graph_scores[graph_id] += insight['match_strength']
        
        question_words = set(question.lower().split())
        for graph in self.insights_data.get('graph_catalog', []):
            graph_words = set(graph['id'].lower().replace('_', ' ').split())
            columns = set([c.lower() for c in graph.get('columns_involved', [])])
            
            word_overlap = len(question_words.intersection(graph_words))
            column_overlap = len(question_words.intersection(columns))
            
            total_overlap = word_overlap + (column_overlap * 2)
            
            if total_overlap > 0:
                graph_scores[graph['id']] += total_overlap * 0.2
        
        related_graphs = []
        for graph_id, score in graph_scores.most_common(Config.TOP_K_GRAPHS):
            graph_detail = self.get_graph_details(graph_id)
            if graph_detail:
                graph_detail['relevance_score'] = float(score)
                graph_detail['linking_method'] = 'heuristic'
                related_graphs.append(graph_detail)
        
        return related_graphs
    
    def get_graph_details(self, graph_id):
        for graph in self.insights_data.get('graph_catalog', []):
            if graph['id'] == graph_id:
                return {
                    "id": graph['id'],
                    "filename": graph['filename'],
                    "type": graph['type'],
                    "path": graph['path'],
                    "format": graph['format'],
                    "task_type": graph.get('task_type'),
                    "columns_involved": graph.get('columns_involved', [])
                }
        return None
    
    def calculate_confidence(self, insights):
        if not insights:
            return "none"
        
        top_strength = insights[0]['match_strength']
        
        if top_strength >= Config.CONFIDENCE_THRESHOLD_HIGH:
            return "high"
        elif top_strength >= Config.CONFIDENCE_THRESHOLD_MEDIUM:
            return "medium"
        else:
            return "low"
    
    def get_response_summary(self, response):
        if response['retrieval_confidence'] == 'none':
            return {
                "answer": "This information is not available in the current analysis.",
                "insights_count": 0,
                "graphs_count": 0
            }
        
        return {
            "answer": response['matched_insights'][0]['insight'] if response['matched_insights'] else "No insights found",
            "insights_count": len(response['matched_insights']),
            "graphs_count": len(response['related_graphs']),
            "all_insights": [i['insight'] for i in response['matched_insights']],
            "all_graphs": [g['filename'] for g in response['related_graphs']]
        }
    
    def interactive_mode(self):
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("Commands: 'quit', 'help', 'context', 'clear'")
        print("="*80 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.save_session()
                    print("\nSession ended")
                    break
                
                if user_input.lower() == 'help':
                    self.show_help()
                    continue
                
                if user_input.lower() == 'context':
                    self.show_context()
                    continue
                
                if user_input.lower() == 'clear':
                    self.session_context = []
                    print("\nContext cleared\n")
                    continue
                
                if not user_input:
                    continue
                
                response = self.retrieve(user_input)
                summary = self.get_response_summary(response)
                
                print(f"\nAssistant: {summary['answer']}")
                
                if summary['insights_count'] > 1:
                    print(f"\nAdditional insights ({summary['insights_count']-1}):")
                    for insight in summary['all_insights'][1:3]:
                        print(f"  - {insight}")
                
                if summary['graphs_count'] > 0:
                    print(f"\nRelated graphs ({summary['graphs_count']}):")
                    for graph in summary['all_graphs']:
                        print(f"  - {graph}")
                
                print(f"\nConfidence: {response['retrieval_confidence']} (match strength, not data certainty)")
                print(f"Matched: {summary['insights_count']} insights, {summary['graphs_count']} graphs")
                
                if response['related_graphs'] and response['related_graphs'][0].get('linking_method') == 'heuristic':
                    print("Note: Graph links are heuristic (keyword-based)")
                
                print()
                
            except KeyboardInterrupt:
                print("\n\nSession ended")
                self.save_session()
                break
            except Exception as e:
                print(f"\nError: {e}")
                continue
    
    def show_help(self):
        print("\nAVAILABLE INFORMATION:")
        print("-" * 80)
        print("Pre-computed insights about:")
        print("  - Dataset statistics")
        print("  - Column summaries")
        print("  - Correlations")
        print("  - Trends (task-aware)")
        print("  - Model performance")
        print("  - Available graphs")
        print("\nExample questions:")
        print("  - How many rows are in the dataset?")
        print("  - What is the mean of [column]?")
        print("  - Are there any correlations?")
        print("  - What trends exist?")
        print("  - Which model performed best?")
        print("  - What graphs show [column]?")
        print("\nNote: I can only retrieve pre-computed information.")
        print("If information was not computed, I will say so.")
        print("-" * 80 + "\n")
    
    def show_context(self):
        print("\nSESSION HISTORY:")
        print("-" * 80)
        if not self.session_context:
            print("No previous questions")
        else:
            for i, item in enumerate(self.session_context[-5:], 1):
                print(f"{i}. {item['question']}")
                print(f"   Retrieval Confidence: {item['response']['retrieval_confidence']}")
        print("-" * 80 + "\n")
    
    def save_session(self):
        if self.session_context:
            session_file = Path(Config.SESSION_FILE)
            with open(session_file, 'w') as f:
                json.dump(self.session_context, f, indent=2)
            print(f"Session saved: {session_file}")
    
    def batch_retrieve(self, questions):
        results = []
        
        print("\nBATCH RETRIEVAL")
        print("="*80)
        
        for i, question in enumerate(questions, 1):
            print(f"\n[{i}/{len(questions)}] {question}")
            response = self.retrieve(question)
            results.append(response)
            
            summary = self.get_response_summary(response)
            print(f"Retrieval Confidence: {response['retrieval_confidence']}")
            print(f"Found: {summary['insights_count']} insights, {summary['graphs_count']} graphs")
        
        return results

class RetrievalValidator:
    
    @staticmethod
    def validate_response(response):
        checks = {
            "has_status": "status" in response,
            "has_question": "question" in response,
            "has_retrieval_confidence": "retrieval_confidence" in response,
            "has_insights": "matched_insights" in response,
            "has_graphs": "related_graphs" in response,
            "has_metadata": "metadata" in response,
            "insights_valid": RetrievalValidator.validate_insights(response.get('matched_insights', [])),
            "graphs_valid": RetrievalValidator.validate_graphs(response.get('related_graphs', []))
        }
        
        return all(checks.values()), checks
    
    @staticmethod
    def validate_insights(insights):
        if not isinstance(insights, list):
            return False
        
        for insight in insights:
            if 'insight' not in insight:
                return False
            if 'match_strength' not in insight:
                return False
            if not isinstance(insight.get('grounding', {}), dict):
                return False
        
        return True
    
    @staticmethod
    def validate_graphs(graphs):
        if not isinstance(graphs, list):
            return False
        
        for graph in graphs:
            if 'id' not in graph:
                return False
            if 'path' not in graph:
                return False
        
        return True

if __name__ == "__main__":
    import sys
    
    retrieval = ChatRetrieval()
    
    try:
        retrieval.setup()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Run insight_extraction.py first")
        sys.exit(1)
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_questions = [
            "How many rows are in the dataset?",
            "What is the average sales?",
            "Are there any correlations?",
            "Which model performed best?",
            "What graphs show profit?"
        ]
        
        results = retrieval.batch_retrieve(test_questions)
        
        print("\n" + "="*80)
        print("VALIDATION")
        print("="*80)
        
        for i, (q, r) in enumerate(zip(test_questions, results), 1):
            valid, checks = RetrievalValidator.validate_response(r)
            print(f"\n{i}. {q}")
            print(f"   Valid: {valid}")
            print(f"   Retrieval Confidence: {r['retrieval_confidence']}")
            print(f"   Insights: {len(r['matched_insights'])}")
            print(f"   Graphs: {len(r['related_graphs'])}")
            if r['related_graphs']:
                print(f"   Graph Linking: {r['related_graphs'][0].get('linking_method', 'unknown')}")
    else:
        retrieval.interactive_mode()