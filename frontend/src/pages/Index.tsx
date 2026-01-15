import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { BarChart3, LineChart, PieChart, TrendingUp, ArrowRight, Sparkles } from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
        <div className="container mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg gradient-accent flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-accent-foreground" />
            </div>
            <span className="font-display text-lg font-bold text-foreground">
              DataViz Pro
            </span>
          </div>
          <nav className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Features
            </a>
            <a href="#pricing" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Pricing
            </a>
            <a href="#docs" className="text-sm text-muted-foreground hover:text-foreground transition-colors">
              Documentation
            </a>
          </nav>
          <div className="flex items-center gap-3">
            <Link to="/auth">
              <Button variant="ghost" size="sm">Sign In</Button>
            </Link>
            <Link to="/auth">
              <Button variant="green"  size="sm">Get Started</Button>
            </Link>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-6">
        <div className="container mx-auto text-center">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 text-accent text-sm font-medium mb-6 animate-fade-in-up">
            <Sparkles className="w-4 h-4" />
            New: AI-Powered Analytics
          </div>
          
          <h1 className="font-display text-4xl sm:text-5xl lg:text-6xl font-bold text-foreground mb-6 animate-fade-in-up animation-delay-100">
            Visualize Your Data
            <br />
            <span className="text-gradient">Like Never Before</span>
          </h1>
          
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto mb-10 animate-fade-in-up animation-delay-200">
            Create stunning charts, interactive dashboards, and powerful analytics 
            with our intuitive platform. Transform raw data into actionable insights.
          </p>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in-up animation-delay-300">
            <Link to="/auth">
              <Button variant="hero" size="lg">
                Start Free Trial
                <ArrowRight className="w-5 h-5" />
              </Button>
            </Link>
            <Button variant="outline" size="lg">
              View Demo
            </Button>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-6 bg-secondary/30">
        <div className="container mx-auto">
          <h2 className="font-display text-3xl font-bold text-center text-foreground mb-12">
            Powerful Features for Data Teams
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: BarChart3,
                title: "Bar & Column Charts",
                description: "Compare categories with beautiful bar visualizations",
                color: "bg-[#3ab7cb]",
              },
              {
                icon: LineChart,
                title: "Line & Area Charts",
                description: "Track trends over time with smooth animations",
                color: "bg-[#f4742b]",
              },
              {
                icon: PieChart,
                title: "Pie & Donut Charts",
                description: "Show proportions with elegant circular displays",
                color: "bg-[#4bb9a4]",
              },
              {
                icon: TrendingUp,
                title: "Real-time Analytics",
                description: "Monitor metrics with live data streaming",
                color: "bg-[#8d6bf7]",
              },
            ].map((feature, index) => (
              <div
                key={feature.title}
                className="bg-card rounded-xl p-6 shadow-soft hover:shadow-lg transition-all duration-300 hover:-translate-y-1 animate-fade-in-up"
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className={`w-12 h-12 rounded-lg ${feature.color} flex items-center justify-center mb-4`}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="font-display font-semibold text-lg text-foreground mb-2">
                  {feature.title}
                </h3>
                <p className="text-sm text-muted-foreground">
                  {feature.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-6">
        <div className="container mx-auto">
          <div className="gradient-primary rounded-2xl p-12 text-center relative overflow-hidden">
            <div className="absolute top-10 left-10 w-64 h-64 bg-accent/20 rounded-full blur-3xl" />
            <div className="absolute bottom-10 right-10 w-96 h-96 bg-chart-purple/20 rounded-full blur-3xl" />
            
            <div className="relative z-10">
              <h2 className="font-display text-3xl sm:text-4xl font-bold text-primary-foreground mb-4">
                Ready to Transform Your Data?
              </h2>
              <p className="text-primary-foreground/80 max-w-xl mx-auto mb-8">
                Join thousands of data professionals who trust DataViz Pro for their analytics needs.
              </p>
              <Link to="/auth">
                <Button variant="green" size="lg">
                  Get Started for Free
                  <ArrowRight className="w-5 h-5" />
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-6 border-t border-border">
        <div className="container mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg gradient-accent flex items-center justify-center">
              <BarChart3 className="w-4 h-4 text-accent-foreground" />
            </div>
            <span className="font-display font-semibold text-foreground">DataViz Pro</span>
          </div>
          <p className="text-sm text-muted-foreground">
            © 2024 DataViz Pro. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
