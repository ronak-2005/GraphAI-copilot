import { useState } from "react";
import { Upload, FileSpreadsheet, Database, Cloud, Link as LinkIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { useNavigate } from "react-router-dom";

export const UploadZone = () => {
  const [isDragging, setIsDragging] = useState(false);
  const navigate = useNavigate();

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    // Simulate file upload and navigate to dashboard
    setTimeout(() => navigate("/dashboard"), 1000);
  };

  const uploadMethods = [
    { icon: FileSpreadsheet, label: "CSV / Excel", desc: "Upload spreadsheets" },
    { icon: Database, label: "Database", desc: "Connect directly" },
    { icon: Cloud, label: "Cloud Storage", desc: "S3, GCS, Azure" },
    { icon: LinkIcon, label: "API", desc: "REST or GraphQL" },
  ];

  return (
    <div className="w-full max-w-4xl mx-auto space-y-8 animate-fade-in-up">
      {/* Main Drop Zone */}
      <Card
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`
          relative overflow-hidden border-2 border-dashed transition-all duration-300
          ${isDragging 
            ? "border-primary bg-primary/5 scale-105 shadow-glow" 
            : "border-border hover:border-primary/50 hover:bg-muted/50"
          }
        `}
      >
        <div className="p-16 text-center space-y-6">
          <div className="mx-auto w-24 h-24 rounded-full bg-gradient-to-br from-primary to-accent flex items-center justify-center animate-float shadow-lg">
            <Upload className="h-12 w-12 text-primary-foreground" />
          </div>
          
          <div className="space-y-2">
            <h3 className="text-2xl font-bold">Drop your data here</h3>
            <p className="text-muted-foreground text-lg">
              or click to browse files
            </p>
          </div>

          <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
            <span>Supports CSV, Excel, JSON</span>
            <span>•</span>
            <span>Up to 100MB</span>
          </div>

          <Button 
            size="lg" 
            className="mt-4 shadow-md hover:shadow-lg transition-smooth"
            onClick={() => navigate("/dashboard")}
          >
            <Upload className="mr-2 h-5 w-5" />
            Choose File
          </Button>
        </div>

        {/* Animated border effect */}
        {isDragging && (
          <div className="absolute inset-0 border-4 border-primary rounded-lg animate-pulse pointer-events-none" />
        )}
      </Card>

      {/* Upload Methods */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {uploadMethods.map((method, index) => (
          <Card
            key={method.label}
            className="p-6 text-center hover:shadow-md hover:-translate-y-1 transition-all duration-300 cursor-pointer border hover:border-primary/50 group"
            style={{ animationDelay: `${index * 100}ms` }}
            onClick={() => navigate("/dashboard")}
          >
            <div className="mx-auto w-12 h-12 rounded-lg bg-muted flex items-center justify-center mb-3 group-hover:bg-primary group-hover:text-primary-foreground transition-smooth">
              <method.icon className="h-6 w-6" />
            </div>
            <h4 className="font-semibold mb-1">{method.label}</h4>
            <p className="text-xs text-muted-foreground">{method.desc}</p>
          </Card>
        ))}
      </div>
    </div>
  );
};
