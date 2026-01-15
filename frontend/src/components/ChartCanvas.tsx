import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ExternalLink, Maximize2, Download } from "lucide-react";

interface Graph {
  id: string;
  filename: string;
  type: string;
  path: string;
  format: string;
  task_type?: string;
  columns_involved?: string[];
}

interface GraphCatalog {
  metadata: {
    generated_at: string;
    task_type: string;
    total_graphs: number;
  };
  graph_catalog: Graph[];
}

export const ChartCanvas = () => {
  const [graphs, setGraphs] = useState<Graph[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedGraph, setSelectedGraph] = useState<Graph | null>(null);

  useEffect(() => {
    loadGraphs();
  }, []);

  const loadGraphs = async () => {
    try {
      const response = await fetch('/api/graphs/catalog');
      
      if (!response.ok) {
        throw new Error('Failed to load graphs');
      }

      const data: GraphCatalog = await response.json();
      setGraphs(data.graph_catalog || []);
      setLoading(false);
    } catch (err) {
      console.error('Error loading graphs:', err);
      setError('Failed to load visualizations');
      setLoading(false);
    }
  };

  const getGraphTypeLabel = (type: string) => {
    const labels: Record<string, string> = {
      'correlation_heatmap': 'Correlation Heatmap',
      'scatter_plot': 'Scatter Plot',
      'bar_chart': 'Bar Chart',
      'pie_chart': 'Pie Chart',
      'distribution_plot': 'Distribution',
      'box_plot': 'Box Plot',
      'violin_plot': 'Violin Plot',
      'confusion_matrix': 'Confusion Matrix',
      'feature_importance': 'Feature Importance',
      'prediction_plot': 'Predictions',
      'error_analysis': 'Error Analysis'
    };
    return labels[type] || type.replace(/_/g, ' ');
  };

  const getGraphColor = (type: string) => {
    const colors: Record<string, string> = {
      'correlation_heatmap': 'border-chart-1',
      'scatter_plot': 'border-chart-2',
      'bar_chart': 'border-chart-3',
      'pie_chart': 'border-chart-4',
      'distribution_plot': 'border-chart-1',
      'feature_importance': 'border-success',
      'confusion_matrix': 'border-warning',
      'prediction_plot': 'border-chart-2',
    };
    return colors[type] || 'border-muted';
  };

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="p-6">
              <Skeleton className="h-8 w-48 mb-4" />
              <Skeleton className="h-64 w-full" />
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <Card className="p-8 text-center">
        <p className="text-destructive mb-4">{error}</p>
        <Button onClick={loadGraphs} variant="outline">
          Retry
        </Button>
      </Card>
    );
  }

  if (graphs.length === 0) {
    return (
      <Card className="p-8 text-center">
        <p className="text-muted-foreground">No visualizations available</p>
        <p className="text-xs text-muted-foreground mt-2">
          Run the graph generation pipeline to create visualizations
        </p>
      </Card>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Data Visualizations</h2>
          <p className="text-sm text-muted-foreground">
            {graphs.length} generated graphs from your data
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={loadGraphs}>
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {graphs.map((graph, index) => (
          <Card
            key={graph.id}
            className={`
              p-6 shadow-md hover:shadow-lg transition-all duration-300
              border-l-4 ${getGraphColor(graph.type)}
              group cursor-pointer
            `}
            style={{ animationDelay: `${index * 50}ms` }}
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold mb-1">
                  {getGraphTypeLabel(graph.type)}
                </h3>
                <p className="text-xs text-muted-foreground">
                  {graph.columns_involved && graph.columns_involved.length > 0
                    ? graph.columns_involved.join(', ')
                    : graph.id.replace(/_/g, ' ')}
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedGraph(graph)}
                  className="h-8 w-8 p-0"
                >
                  <Maximize2 className="h-4 w-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  asChild
                  className="h-8 w-8 p-0"
                >
                  <a href={`/api/graphs/${graph.filename}`} download>
                    <Download className="h-4 w-4" />
                  </a>
                </Button>
              </div>
            </div>

            <div className="relative bg-muted/30 rounded-lg overflow-hidden">
              {graph.format === 'html' ? (
                <iframe
                  src={`/api/graphs/${graph.filename}`}
                  className="w-full h-80 border-0"
                  title={graph.id}
                  sandbox="allow-scripts allow-same-origin"
                />
              ) : (
                <img
                  src={`/api/graphs/${graph.filename}`}
                  alt={graph.id}
                  className="w-full h-80 object-contain"
                />
              )}
            </div>

            <div className="mt-4 flex items-center justify-between">
              <span className="text-xs text-muted-foreground capitalize">
                {graph.format} • {graph.task_type || 'analysis'}
              </span>
              <Button
                variant="ghost"
                size="sm"
                asChild
                className="h-8 text-xs"
              >
                <a
                  href={`/api/graphs/${graph.filename}`}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Open Full View
                  <ExternalLink className="ml-2 h-3 w-3" />
                </a>
              </Button>
            </div>
          </Card>
        ))}
      </div>

      {selectedGraph && (
        <GraphModal
          graph={selectedGraph}
          onClose={() => setSelectedGraph(null)}
        />
      )}
    </div>
  );
};

const GraphModal = ({ graph, onClose }: { graph: Graph; onClose: () => void }) => {
  return (
    <div
      className="fixed inset-0 bg-background/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <Card
        className="w-full max-w-6xl max-h-[90vh] overflow-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="p-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold">{graph.id.replace(/_/g, ' ')}</h2>
              <p className="text-sm text-muted-foreground">
                {graph.type.replace(/_/g, ' ')}
              </p>
            </div>
            <Button variant="ghost" onClick={onClose}>
              Close
            </Button>
          </div>

          <div className="bg-muted/30 rounded-lg overflow-hidden">
            {graph.format === 'html' ? (
              <iframe
                src={`/api/graphs/${graph.filename}`}
                className="w-full h-[70vh] border-0"
                title={graph.id}
                sandbox="allow-scripts allow-same-origin"
              />
            ) : (
              <img
                src={`/api/graphs/${graph.filename}`}
                alt={graph.id}
                className="w-full h-auto"
              />
            )}
          </div>
        </div>
      </Card>
    </div>
  );
};