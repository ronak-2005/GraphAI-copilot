"""
ROLE: Analytical Truth Generator
Purpose:
- Extract grounded, explainable insights from data, models, and graphs
- No LLM, no chat, no vision logic
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class Config:
    DATA_FOLDER = "../ml/ready_data/"
    GRAPHS_FOLDER = "../ml/graph_output/"
    INSIGHTS_FOLDER = "../ml/graph_insights/"
    
    TRAIN_DATA = "../ml/ready_data/train.csv"
    TEST_DATA = "../ml/ready_data/test.csv"
    ORIGINAL_DATA = "../ml/moretoo/Ecommerce/train/cleaned_Customers.csv"
    
    MODEL_RESULTS = "../ml/model_results/"
    METADATA_FILE = "../ml/artifacts/metadata.json"
    
    GENERATE_EMBEDDINGS = False #on krna hai jab app live hoga

class InsightExtractor:
    
    def __init__(self):
        self.insights = []
        self.graph_catalog = []
        self.data_summary = {}
        self.task_type = None
        self.target_column = None
        
    def run(self):
        print("="*80)
        print("GRAPH INSIGHT EXTRACTION PIPELINE")
        print("="*80)
        
        Path(Config.INSIGHTS_FOLDER).mkdir(parents=True, exist_ok=True)
        
        self.load_metadata()
        self.load_data()
        self.extract_dataset_insights()
        self.extract_statistical_insights()
        self.extract_model_insights()
        self.catalog_graphs()
        self.generate_structured_insights()
        self.link_insights_to_graphs()
        self.save_insights()
        self.generate_embeddings()
        self.create_search_index()
        
        print("\n" + "="*80)
        print("INSIGHT EXTRACTION COMPLETED")
        print("="*80)
    
    def load_metadata(self):
        print("\nSTEP 1: Loading Task Metadata")
        
        try:
            metadata_path = Path(Config.METADATA_FILE)
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.task_type = metadata.get('task_type', 'unknown')
                    self.target_column = metadata.get('target_column', 'unknown')
                print(f"   Task Type: {self.task_type}")
                print(f"   Target Column: {self.target_column}")
            else:
                print("   Metadata not found, will auto-detect")
        except Exception as e:
            print(f"   Warning: {e}")
    
    def load_data(self):
        print("\nSTEP 2: Loading Data")
        
        try:
            self.df_original = pd.read_csv(Config.ORIGINAL_DATA)
            print(f"   Original data: {self.df_original.shape}")
        except:
            print("   Original data not found")
            self.df_original = None
        
        try:
            self.df_train = pd.read_csv(Config.TRAIN_DATA)
            self.df_test = pd.read_csv(Config.TEST_DATA)
            print(f"   Train data: {self.df_train.shape}")
            print(f"   Test data: {self.df_test.shape}")
            
            if self.task_type is None:
                if self.target_column and self.target_column in self.df_train.columns:
                    unique_vals = self.df_train[self.target_column].nunique()
                    self.task_type = "classification" if unique_vals < 50 else "regression"
                    print(f"   Auto-detected task: {self.task_type}")
        except:
            print("   ML-ready data not found")
            self.df_train = None
            self.df_test = None
    
    def extract_dataset_insights(self):
        print("\nSTEP 3: Extracting Dataset Insights")
        
        if self.df_original is not None:
            df = self.df_original
        elif self.df_train is not None:
            df = self.df_train
        else:
            print("   No data available")
            return
        
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        self.data_summary = {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "dataset_info": {
                "total_rows": int(len(df)),
                "total_columns": int(len(df.columns)),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "missing_values": int(df.isnull().sum().sum()),
                "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2)
            },
            "columns": {}
        }
        
        for col in numeric_cols:
            stats = df[col].describe()
            self.data_summary["columns"][col] = {
                "type": "numeric",
                "count": int(stats['count']),
                "mean": float(stats['mean']) if not np.isnan(stats['mean']) else None,
                "std": float(stats['std']) if not np.isnan(stats['std']) else None,
                "min": float(stats['min']) if not np.isnan(stats['min']) else None,
                "max": float(stats['max']) if not np.isnan(stats['max']) else None,
                "median": float(df[col].median()) if not np.isnan(df[col].median()) else None,
                "missing": int(df[col].isnull().sum()),
                "missing_pct": float(df[col].isnull().sum() / len(df) * 100),
                "quartiles": {
                    "q1": float(stats['25%']) if not np.isnan(stats['25%']) else None,
                    "q3": float(stats['75%']) if not np.isnan(stats['75%']) else None
                }
            }
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            self.data_summary["columns"][col] = {
                "type": "categorical",
                "unique_values": int(df[col].nunique()),
                "most_common": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_common_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                "most_common_pct": float(value_counts.iloc[0] / len(df) * 100) if len(value_counts) > 0 else None,
                "missing": int(df[col].isnull().sum()),
                "missing_pct": float(df[col].isnull().sum() / len(df) * 100),
                "top_5_values": {str(k): int(v) for k, v in value_counts.head(5).to_dict().items()},
                "distribution": {str(k): int(v) for k, v in value_counts.to_dict().items()}
            }
        
        print(f"   Extracted insights for {len(df.columns)} columns")
    
    def extract_statistical_insights(self):
        print("\nSTEP 4: Extracting Statistical Insights")
        
        if self.df_original is not None:
            df = self.df_original
        elif self.df_train is not None:
            df = self.df_train
        else:
            return
        
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7 and not np.isnan(corr_value):
                        strong_correlations.append({
                            "feature_1": corr_matrix.columns[i],
                            "feature_2": corr_matrix.columns[j],
                            "correlation": float(corr_value),
                            "strength": "strong_positive" if corr_value > 0 else "strong_negative",
                            "metric": "correlation",
                            "operation": "pearson"
                        })
            
            self.data_summary["correlations"] = strong_correlations
            print(f"   Found {len(strong_correlations)} strong correlations")
        
        trends = []
        if self.task_type == "regression":
            for col in numeric_cols[:5]:
                values = df[col].dropna()
                if len(values) > 10:
                    first_half = values[:len(values)//2].mean()
                    second_half = values[len(values)//2:].mean()
                    pct_change = ((second_half - first_half) / first_half * 100) if first_half != 0 else 0
                    
                    if abs(pct_change) > 5:
                        trends.append({
                            "column": col,
                            "trend": "increasing" if pct_change > 0 else "decreasing",
                            "percentage_change": float(pct_change),
                            "metric": col,
                            "operation": "trend_analysis",
                            "time_window": "full_dataset",
                            "task_aware": True,
                            "applicable_to": "regression"
                        })
        elif self.task_type == "classification":
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols[:5]:
                value_counts = df[col].value_counts()
                if len(value_counts) > 1:
                    top_category = value_counts.index[0]
                    top_pct = (value_counts.iloc[0] / len(df)) * 100
                    
                    trends.append({
                        "column": col,
                        "trend": "dominant_category",
                        "dominant_value": str(top_category),
                        "dominance_pct": float(top_pct),
                        "metric": col,
                        "dimension": "categorical_distribution",
                        "operation": "frequency_analysis",
                        "task_aware": True,
                        "applicable_to": "classification"
                    })
        
        self.data_summary["trends"] = trends
        print(f"   Detected {len(trends)} task-aware patterns")
    
    def extract_model_insights(self):
        print("\nSTEP 5: Extracting Model Performance Insights")
        
        try:
            comparison_file = Path(Config.MODEL_RESULTS) / "comparison.csv"
            if comparison_file.exists():
                comparison = pd.read_csv(comparison_file, index_col=0)
                
                self.data_summary["model_performance"] = {
                    "task_type": self.task_type,
                    "models_trained": list(comparison.index),
                    "best_model": comparison.index[0],
                    "metrics": comparison.to_dict('index'),
                    "metric_type": "classification_metrics" if self.task_type == "classification" else "regression_metrics"
                }
                
                print(f"   Extracted insights for {len(comparison)} models")
            else:
                print("   Model comparison file not found")
        except Exception as e:
            print(f"   Could not extract model insights: {e}")
    
    def catalog_graphs(self):
        print("\nSTEP 6: Cataloging Generated Graphs")
        
        graphs_path = Path(Config.GRAPHS_FOLDER)
        if not graphs_path.exists():
            print("   Graphs folder not found")
            return
        
        graph_files = list(graphs_path.glob("*.html")) + list(graphs_path.glob("*.png"))
        
        for graph_file in graph_files:
            graph_type = self.infer_graph_type(graph_file.name) #todo V3
            columns_involved = self.extract_columns_from_filename(graph_file.name)
            
            graph_entry = {
                "id": graph_file.stem,
                "filename": graph_file.name,
                "type": graph_type,
                "path": str(graph_file),
                "format": graph_file.suffix[1:],
                "created": datetime.fromtimestamp(graph_file.stat().st_mtime).isoformat(),
                "columns_involved": columns_involved,
                "task_type": self.infer_graph_task_relevance(graph_type)
            }
            
            self.graph_catalog.append(graph_entry)
        
        print(f"   Cataloged {len(self.graph_catalog)} graphs")
    
    def infer_graph_type(self, filename):
        filename_lower = filename.lower()
        
        type_mapping = {
            'correlation': 'correlation_heatmap',
            'heatmap': 'correlation_heatmap',
            'scatter': 'scatter_plot',
            'bar': 'bar_chart',
            'categorical': 'bar_chart',
            'pie': 'pie_chart',
            'distribution': 'distribution_plot',
            'histogram': 'distribution_plot',
            'box': 'box_plot',
            'violin': 'violin_plot',
            'confusion': 'confusion_matrix',
            'importance': 'feature_importance',
            'prediction': 'prediction_plot',
            'error': 'error_analysis'
        }
        
        for keyword, graph_type in type_mapping.items():
            if keyword in filename_lower:
                return graph_type
        
        return "unknown"
    
    def extract_columns_from_filename(self, filename):
        columns = []
        parts = filename.replace('.html', '').replace('.png', '').split('_')
        
        for col_info in self.data_summary.get('columns', {}).keys():
            if any(col_info.lower() in part.lower() for part in parts):
                columns.append(col_info)
        
        return columns
    
    def infer_graph_task_relevance(self, graph_type):
        regression_graphs = ['scatter_plot', 'prediction_plot', 'error_analysis', 'distribution_plot']
        classification_graphs = ['confusion_matrix', 'bar_chart', 'pie_chart']
        both = ['correlation_heatmap', 'feature_importance', 'box_plot', 'violin_plot']
        
        if graph_type in regression_graphs:
            return 'regression'
        elif graph_type in classification_graphs:
            return 'classification'
        elif graph_type in both:
            return 'both'
        return 'unknown'
    
    def generate_structured_insights(self):
        print("\nSTEP 7: Generating Structured Insights")
        
        insights = []
        
        if "dataset_info" in self.data_summary:
            info = self.data_summary["dataset_info"]
            insights.append({
                "id": "dataset_overview",
                "category": "dataset_overview",
                "insight": f"The dataset contains {info['total_rows']:,} rows and {info['total_columns']} columns with {info['numeric_columns']} numeric and {info['categorical_columns']} categorical features.",
                "importance": "high",
                "task_type": self.task_type,
                "grounding": {
                    "metric": "dataset_size",
                    "operation": "count",
                    "values": {
                        "rows": info['total_rows'],
                        "columns": info['total_columns']
                    }
                },
                "supports_graphs": ["overview"]
            })
        
        if "columns" in self.data_summary:
            for col, stats in self.data_summary["columns"].items():
                if stats["type"] == "numeric":
                    if stats["mean"] is not None:
                        insights.append({
                            "id": f"numeric_{col}",
                            "category": "column_statistics",
                            "column": col,
                            "insight": f"{col} has a mean of {stats['mean']:.2f}, ranging from {stats['min']:.2f} to {stats['max']:.2f}.",
                            "importance": "medium",
                            "task_type": self.task_type,
                            "grounding": {
                                "metric": col,
                                "dimension": "distribution",
                                "operation": "descriptive_statistics",
                                "values": {
                                    "mean": stats['mean'],
                                    "min": stats['min'],
                                    "max": stats['max'],
                                    "median": stats['median']
                                }
                            },
                            "supports_graphs": self.find_graphs_for_column(col)
                        })
                elif stats["type"] == "categorical":
                    insights.append({
                        "id": f"categorical_{col}",
                        "category": "column_statistics",
                        "column": col,
                        "insight": f"{col} has {stats['unique_values']} unique values, with '{stats['most_common']}' being the most common ({stats['most_common_pct']:.1f}%).",
                        "importance": "medium",
                        "task_type": self.task_type,
                        "grounding": {
                            "metric": col,
                            "dimension": "categorical_distribution",
                            "operation": "frequency_count",
                            "values": {
                                "unique_count": stats['unique_values'],
                                "mode": stats['most_common'],
                                "mode_frequency": stats['most_common_count'],
                                "mode_percentage": stats['most_common_pct']
                            }
                        },
                        "supports_graphs": self.find_graphs_for_column(col)
                    })
        
        if "correlations" in self.data_summary:
            for corr in self.data_summary["correlations"][:5]:
                insights.append({
                    "id": f"corr_{corr['feature_1']}_{corr['feature_2']}",
                    "category": "correlation",
                    "insight": f"{corr['feature_1']} and {corr['feature_2']} show a {corr['strength']} correlation ({corr['correlation']:.2f}).",
                    "importance": "high",
                    "task_type": self.task_type,
                    "grounding": {
                        "metric": "correlation",
                        "dimension": f"{corr['feature_1']}_vs_{corr['feature_2']}",
                        "operation": "pearson_correlation",
                        "values": {
                            "correlation_coefficient": corr['correlation'],
                            "strength": corr['strength']
                        }
                    },
                    "supports_graphs": ["correlation_heatmap", "scatter_matrix"]
                })
        
        if "trends" in self.data_summary:
            for trend in self.data_summary["trends"]:
                if trend.get("applicable_to") == self.task_type or trend.get("applicable_to") == "both":
                    if self.task_type == "regression":
                        insights.append({
                            "id": f"trend_{trend['column']}",
                            "category": "trend",
                            "insight": f"{trend['column']} shows a {trend['trend']} trend with {abs(trend['percentage_change']):.1f}% change.",
                            "importance": "high",
                            "task_type": "regression",
                            "grounding": trend,
                            "supports_graphs": self.find_graphs_for_column(trend['column'])
                        })
                    else:
                        insights.append({
                            "id": f"distribution_{trend['column']}",
                            "category": "distribution",
                            "insight": f"{trend['column']} is dominated by '{trend['dominant_value']}' at {trend['dominance_pct']:.1f}%.",
                            "importance": "high",
                            "task_type": "classification",
                            "grounding": trend,
                            "supports_graphs": self.find_graphs_for_column(trend['column'])
                        })
        
        if "model_performance" in self.data_summary:
            model_info = self.data_summary["model_performance"]
            insights.append({
                "id": "model_best",
                "category": "model_performance",
                "insight": f"Best performing model is {model_info['best_model']} among {len(model_info['models_trained'])} trained models.",
                "importance": "high",
                "task_type": self.task_type,
                "grounding": {
                    "metric": "model_performance",
                    "operation": "comparison",
                    "values": {
                        "best_model": model_info['best_model'],
                        "total_models": len(model_info['models_trained'])
                    }
                },
                "supports_graphs": ["model_comparison", "confusion_matrix", "predictions"]
            })
        
        self.insights = insights
        print(f"   Generated {len(insights)} structured insights")
    
    def find_graphs_for_column(self, column):
        graphs = []
        for graph in self.graph_catalog:
            if column in graph.get('columns_involved', []) or column.lower() in graph['id'].lower():
                graphs.append(graph['id'])
        return graphs
    
    def link_insights_to_graphs(self):
        print("\nSTEP 8: Linking Insights to Graphs")
        
        links_created = 0
        for insight in self.insights:
            if not insight.get('supports_graphs'):
                insight['supports_graphs'] = []
                
                if insight.get('column'):
                    insight['supports_graphs'] = self.find_graphs_for_column(insight['column'])
                
                if insight['category'] == 'correlation':
                    insight['supports_graphs'].extend(['correlation_heatmap', 'scatter_matrix'])
                
                if insight['category'] == 'model_performance':
                    insight['supports_graphs'].extend(['model_comparison', 'feature_importance'])
                
                insight['supports_graphs'] = list(set(insight['supports_graphs']))
                
                if insight['supports_graphs']:
                    links_created += 1
        
        print(f"   Created {links_created} insight-to-graph links")
    
    def save_insights(self):
        print("\nSTEP 9: Saving Insights")
        
        master_insights = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "2.0",
                "task_type": self.task_type,
                "target_column": self.target_column,
                "total_insights": len(self.insights),
                "total_graphs": len(self.graph_catalog)
            },
            "data_summary": self.data_summary,
            "graph_catalog": self.graph_catalog,
            "insights": self.insights
        }
        
        output_file = Path(Config.INSIGHTS_FOLDER) / "master_insights.json"
        with open(output_file, 'w') as f:
            json.dump(master_insights, f, indent=2, default=str)
        
        print(f"   Saved master insights to: {output_file}")
        
        insights_only_file = Path(Config.INSIGHTS_FOLDER) / "insights_only.json"
        with open(insights_only_file, 'w') as f:
            json.dump(self.insights, f, indent=2)
        
        print(f"   Saved insights only to: {insights_only_file}")
        
        readable_file = Path(Config.INSIGHTS_FOLDER) / "insights_readable.txt"
        with open(readable_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AUTOMATED DATA INSIGHTS\n")
            f.write(f"Task Type: {self.task_type}\n")
            f.write(f"Target: {self.target_column}\n")
            f.write("="*80 + "\n\n")
            
            categories = {}
            for insight in self.insights:
                cat = insight.get("category", "other")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(insight)
            
            for category, items in categories.items():
                f.write(f"\n{category.upper().replace('_', ' ')}\n")
                f.write("-" * 80 + "\n")
                for item in items:
                    f.write(f"  - {item['insight']}\n")
                    if item.get('supports_graphs'):
                        f.write(f"    Graphs: {', '.join(item['supports_graphs'][:3])}\n")
        
        print(f"   Saved readable insights to: {readable_file}")
    
    def generate_embeddings(self):
        print("\nSTEP 10: Generating Embeddings")
        
        if not Config.GENERATE_EMBEDDINGS:
            print("   Embedding generation disabled")
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            texts = [insight['insight'] for insight in self.insights]
            embeddings = model.encode(texts)
            
            embeddings_data = []
            for i, (insight, embedding) in enumerate(zip(self.insights, embeddings)):
                embeddings_data.append({
                    "id": insight.get('id', i),
                    "text": insight['insight'],
                    "category": insight.get('category', 'unknown'),
                    "importance": insight.get('importance', 'medium'),
                    "task_type": insight.get('task_type'),
                    "supports_graphs": insight.get('supports_graphs', []),
                    "embedding": embedding.tolist()
                })
            
            embeddings_file = Path(Config.INSIGHTS_FOLDER) / "embeddings.json"
            with open(embeddings_file, 'w') as f:
                json.dump(embeddings_data, f, indent=2)
            
            print(f"   Generated embeddings for {len(embeddings_data)} insights")
            print(f"   Saved to: {embeddings_file}")
            
        except ImportError:
            print("   sentence-transformers not installed")
            print("   Install with: pip install sentence-transformers")
        except Exception as e:
            print(f"   Error generating embeddings: {e}")
    
    def create_search_index(self):
        print("\nSTEP 11: Creating Search Index")
        
        search_index = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "task_type": self.task_type,
                "total_entries": len(self.insights) + len(self.graph_catalog)
            },
            "searchable_content": []
        }
        
        for insight in self.insights:
            search_index["searchable_content"].append({
                "type": "insight",
                "id": insight.get('id', f"insight_{self.insights.index(insight)}"),
                "category": insight.get('category', 'general'),
                "content": insight['insight'],
                "importance": insight.get('importance', 'medium'),
                "task_type": insight.get('task_type'),
                "keywords": self.extract_keywords_freq(insight['insight']),
                "supports_graphs": insight.get('supports_graphs', []),
                "grounding": insight.get('grounding', {})
            })
        
        for graph in self.graph_catalog:
            search_index["searchable_content"].append({
                "type": "graph",
                "id": graph['id'],
                "category": graph['type'],
                "content": f"{graph['type'].replace('_', ' ')} showing {graph['id'].replace('_', ' ')}",
                "path": graph['path'],
                "task_type": graph.get('task_type'),
                "columns_involved": graph.get('columns_involved', []),
                "keywords": self.extract_keywords_freq(graph['id'])
            })
        
        index_file = Path(Config.INSIGHTS_FOLDER) / "search_index.json"
        with open(index_file, 'w') as f:
            json.dump(search_index, f, indent=2)
        
        print(f"   Created search index with {len(search_index['searchable_content'])} entries")
        print(f"   Saved to: {index_file}")
    
    def extract_keywords_freq(self, text):
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                       'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                       'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                       'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'show',
                       'shows', 'showing', 'contains', 'has'}
        
        words = text.lower().split()
        words = [w.strip('.,!?;:') for w in words if w not in common_words and len(w) > 2]
        
        word_counts = Counter(words)
        
        keywords = []
        for word, count in word_counts.most_common(10):
            if len(word) > 3:
                keywords.append(word)
        
        return keywords

if __name__ == "__main__":
    extractor = InsightExtractor()
    extractor.run()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Task Type: {extractor.task_type}")
    print(f"Target Column: {extractor.target_column}")
    print(f"Total insights: {len(extractor.insights)}")
    print(f"Graphs cataloged: {len(extractor.graph_catalog)}")
    print(f"\nOutput files:")
    print(f"  - master_insights.json (complete structured data)")
    print(f"  - insights_only.json (insights array)")
    print(f"  - insights_readable.txt (human-readable)")
    print(f"  - search_index.json (searchable index with grounding)")
    print(f"  - embeddings.json (vector embeddings)")
    print("="*80)