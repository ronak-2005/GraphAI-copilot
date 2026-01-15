import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  User,
  LogOut,
  Sparkles,
  Send,
  TrendingUp,
  AlertCircle,
  Lightbulb,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export const ChatSidebar = () => {
  const navigate = useNavigate();
  const { setUser } = useAuth();

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setUser(null);
    navigate("/auth");
  };

  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content:
        "I am your AI data analyst. Ask about trends, performance, anomalies, or insights derived from your data.",
    },
  ]);

  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  const suggestedQuestions = [
    { icon: TrendingUp, text: "Show trends over time", color: "text-chart-1" },
    { icon: AlertCircle, text: "Find anomalies", color: "text-chart-5" },
    { icon: Lightbulb, text: "Suggest improvements", color: "text-chart-4" },
  ];

  const handleSend = () => {
    if (!input.trim()) return;

    setMessages((prev) => [...prev, { role: "user", content: input }]);
    setInput("");
    setIsTyping(true);

    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content:
            "Based on the available insights, sales increased in recent periods. Would you like a breakdown by category or region?",
        },
      ]);
      setIsTyping(false);
    }, 1200);
  };

  return (
    <Card className="flex flex-col min-h-full shadow-lg border-l">

      {/* Header */}
      <div className="px-4 pt-4 pb-3 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-7 w-7 items-center justify-center rounded-md bg-gradient-to-br from-primary to-accent">
              <Sparkles className="h-4 w-4 text-primary-foreground" />
            </div>
            <div>
              <h3 className="text-sm font-semibold leading-tight">
                AI Assistant
              </h3>
              <p className="text-[11px] text-muted-foreground">
                Insight retrieval mode
              </p>
            </div>
          </div>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon">
                <User className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44">
              <DropdownMenuItem>
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                className="text-destructive"
                onClick={handleLogout}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 min-h-0 px-4 py-3">
        <div className="space-y-3">
          {messages.map((message, index) => {
            const isIntro = index === 0 && message.role === "assistant";

            return (
              <div
                key={index}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`
                    max-w-[85%] rounded-lg px-3 py-1.5 text-xs leading-snug
                    ${
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : isIntro
                        ? "bg-muted/40 border text-muted-foreground"
                        : "bg-muted"
                    }
                  `}
                >
                  {message.content}
                </div>
              </div>
            );
          })}

          {isTyping && (
            <div className="flex justify-start">
              <div className="bg-muted rounded-lg px-3 py-2">
                <div className="flex gap-1">
                  <span className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse" />
                  <span
                    className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse"
                    style={{ animationDelay: "0.15s" }}
                  />
                  <span
                    className="w-2 h-2 rounded-full bg-muted-foreground animate-pulse"
                    style={{ animationDelay: "0.3s" }}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Suggested Questions */}
      <div className="px-4 py-3 border-t bg-muted/30">
        <p className="text-[11px] font-medium text-muted-foreground mb-2">
          Suggested questions
        </p>
        <div className="space-y-1">
          {suggestedQuestions.map((q, index) => (
            <Button
              key={index}
              variant="ghost"
              size="sm"
              className="w-full justify-start text-xs py-1.5"
              onClick={() => setInput(q.text)}
            >
              <q.icon className={`mr-2 h-4 w-4 ${q.color}`} />
              {q.text}
            </Button>
          ))}
        </div>
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t">
        <div className="flex gap-2">
          <Input
            placeholder="Ask about your data..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            className="flex-1"
          />
          <Button size="icon" onClick={handleSend}>
            <Send className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </Card>
  );
};
