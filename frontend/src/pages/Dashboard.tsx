import { useState } from "react";
import { ChartCanvas } from "@/components/ChartCanvas";
import { ChatSidebar } from "@/components/ChatSidebar";
import { InsightsPanel } from "@/components/InsightsPanel";
import { Switch } from "@/components/ui/switch";
import { BarChart3 } from "lucide-react";
import { Link } from "react-router-dom";

type ViewMode = "graphs" | "insights";

const Dashboard = () => {
  const [viewMode, setViewMode] = useState<ViewMode>("graphs");

  return (
    <div className="h-screen flex flex-col bg-background overflow-hidden">

      {/* ───────────── Header (FULL WIDTH, NO GAP) ───────────── */}
      <header className="border-b bg-background">
        <div className="h-16 flex items-center justify-between px-6">

          {/* Left: Logo + Title */}
          <div className="flex items-center space-x-8">
            <Link to="/" className="flex items-center space-x-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-accent">
                <BarChart3 className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                DataViz AI
              </span>
            </Link>

            <div className="hidden md:block">
              <h2 className="text-sm font-semibold">
                {viewMode === "graphs"
                  ? "Business Performance Overview"
                  : "ML-Detected Insights"}
              </h2>
              <p className="text-xs text-muted-foreground">
                {viewMode === "graphs"
                  ? "Interactive data visualizations"
                  : "Pre-computed analytical intelligence"}
              </p>
            </div>
          </div>

          {/* Right: Toggle */}
          <div className="flex items-center gap-3">
            <span
              className={`text-xs ${viewMode === "graphs"
                  ? "text-foreground font-medium"
                  : "text-muted-foreground"
                }`}
            >
              Graphs
            </span>

            <Switch
              checked={viewMode === "insights"}
              onCheckedChange={(checked) =>
                setViewMode(checked ? "insights" : "graphs")
              }
            />

            <span
              className={`text-xs ${viewMode === "insights"
                  ? "text-foreground font-medium"
                  : "text-muted-foreground"
                }`}
            >
              AI Insights
            </span>
          </div>
        </div>
      </header>

      {/* ───────────── Main Layout ───────────── */}
      <div className="flex-1 flex overflow-hidden">

        {/* Main Content */}
        <main className="flex-1 overflow-hidden">
          <div className="h-full overflow-auto p-6">
            {viewMode === "graphs" ? (
              <ChartCanvas />
            ) : (
              <div className="max-w-6xl mx-auto">
                <InsightsPanel />
              </div>
            )}
          </div>
        </main>

        {/* Chat Sidebar (FULL HEIGHT) */}
        <aside className="w-[420px] flex-shrink-0">
          <ChatSidebar />
        </aside>
      </div>
    </div>
  );
};

export default Dashboard;
