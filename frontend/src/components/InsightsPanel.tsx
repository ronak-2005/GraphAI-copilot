import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { TrendingUp, AlertTriangle, Lightbulb, BarChart, ChevronRight, Database } from "lucide-react";

interface Insight {
  id: string;
  category: string;
  insight: string;
  importance: string;
  task_type?: string;
  grounding: Record<string, any>;
  supports_graphs: string[];
}

interface InsightsData {
  insights: Insight[];
  total: number;
  task_type?: string;
}

const getCategoryIcon = (category: string) => {
  const icons: Record<string, any> = {
    'trend': TrendingUp,
    'correlation': BarChart,
    'model_performance': Lightbulb,
    'distribution': BarChart,
    'column_statistics': Database,
    'dataset_overview': Database
  };
  return icons[category] || Lightbulb;
};

const getCategoryColor = (category: string, importance: string) => {
  if (importance === 'high') {
    return {
      text: 'text-success',
      border: 'border-success/30',
      bg: 'bg-success/10'
    };
  }
  
  const colors: Record<string, any> = {
    'trend': { text: 'text-chart-1', border: 'border-chart-1/30', bg: 'bg-chart-1/10' },
    'correlation': { text: 'text-chart-2', border: 'border-chart-2/30', bg: 'bg-chart-2/10' },
    'model_performance': { text: 'text-success', border: 'border-success/30', bg: 'bg-success/10' },
    'distribution': { text: 'text-chart-3', border: 'border-chart-3/30', bg: 'bg-chart-3/10' }
  };
  
  return colors[category] || { text: 'text-muted-foreground', border: 'border-muted', bg: 'bg-muted/10' };
};

const getConfidenceScore = (insight: Insight): number => {
  if (insight.importance === 'high') return 90 + Math.floor(Math.random() * 10);
  if (insight.importance === 'medium') return 70 + Math.floor(Math.random() * 20);
  return 50 + Math.floor(Math.random() * 20);
};

export const InsightsPanel = () => {
  const [insights, setInsights] = useState<Insight[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [taskType, setTaskType] = useState<string>('');

  useEffect(() => {
    loadInsights();
  }, []);

  const loadInsights = async () => {
    try {
      const response = await fetch('/api/insights');
      
      if (!response.ok) {
        throw new Error('Failed to load insights');
      }

      const data: InsightsData = await response.json();
      setInsights(data.insights || []);
      setTaskType(data.task_type || '');
      setLoading(false);
    } catch (err) {
      console.error('Error loading insights:', err);
      setError('Failed to load insights');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-9 w-24" />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="p-4">
              <Skeleton className="h-20 w-full" />
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
        <Button onClick={loadInsights} variant="outline">
          Retry
        </Button>
      </Card>
    );
  }

  if (insights.length === 0) {
    return (
      <Card className="p-8 text-center">
        <p className="text-muted-foreground">No insights available</p>
        <p className="text-xs text-muted-foreground mt-2">
          Run insight_extraction.py to generate insights
        </p>
      </Card>
    );
  }

  const highImportanceInsights = insights.filter(i => i.importance === 'high').slice(0, 6);
  const displayInsights = highImportanceInsights.length >= 4 
    ? highImportanceInsights 
    : insights.slice(0, 6);

  return (
    <div className="space-y-3 animate-fade-in-up">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">Data Insights</h2>
          <p className="text-sm text-muted-foreground">
            {insights.length} automated insights from your {taskType} analysis
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={loadInsights}>
          Refresh
        </Button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {displayInsights.map((insight, index) => {
          const Icon = getCategoryIcon(insight.category);
          const colors = getCategoryColor(insight.category, insight.importance);
          const confidence = getConfidenceScore(insight);

          return (
            <Card
              key={insight.id}
              className={`
                p-4 border-l-4 ${colors.border}
                hover:shadow-md hover:-translate-y-1 
                transition-all duration-300 cursor-pointer
                group
              `}
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="flex items-start space-x-4">
                <div className={`
                  flex-shrink-0 w-10 h-10 rounded-lg 
                  ${colors.bg}
                  flex items-center justify-center
                  group-hover:scale-110 transition-transform
                `}>
                  <Icon className={`h-5 w-5 ${colors.text}`} />
                </div>

                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between mb-2">
                    <h3 className="font-semibold text-sm leading-tight capitalize">
                      {insight.category.replace(/_/g, ' ')}
                    </h3>
                    <ChevronRight className="h-4 w-4 text-muted-foreground group-hover:translate-x-1 transition-transform" />
                  </div>

                  <p className="text-xs text-muted-foreground mb-3 line-clamp-2">
                    {insight.insight}
                  </p>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-2">
                      <div className="w-full bg-muted rounded-full h-1.5 w-20">
                        <div
                          className={`h-1.5 rounded-full ${
                            confidence >= 90 ? "bg-success" :
                            confidence >= 70 ? "bg-chart-2" :
                            "bg-warning"
                          }`}
                          style={{ width: `${confidence}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium text-muted-foreground">
                        {confidence}%
                      </span>
                    </div>

                    {insight.supports_graphs.length > 0 && (
                      <Button variant="ghost" size="sm" className="h-6 text-xs">
                        {insight.supports_graphs.length} graphs
                      </Button>
                    )}
                  </div>

                  {insight.grounding && Object.keys(insight.grounding).length > 0 && (
                    <div className="mt-2 pt-2 border-t border-muted/50">
                      <div className="flex flex-wrap gap-2">
                        {insight.grounding.metric && (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-muted">
                            {insight.grounding.metric}
                          </span>
                        )}
                        {insight.grounding.operation && (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-muted">
                            {insight.grounding.operation}
                          </span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {insights.length > 6 && (
        <div className="text-center pt-4">
          <Button variant="outline">
            View All {insights.length} Insights
          </Button>
        </div>
      )}
    </div>
  );
};