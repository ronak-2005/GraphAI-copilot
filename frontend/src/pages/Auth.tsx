// src/pages/Auth.tsx
import { useState } from "react";
import { useAuth } from "@/hooks/useAuth"; // your context
import { AuthService } from "@/services/auth/authService";

import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import DataViz  from "@/components/DataViz";
import { useToast } from "@/components/ui/use-toast";
import { BarChart3, Eye, EyeOff, Mail, Lock, User, ArrowRight } from "lucide-react";

const Auth = () => {
  const { user , setUser } = useAuth();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [isLogin, setIsLogin] = useState(true);
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);

  const [formData, setFormData] = useState({
    email: "",
    password: "",
    name: "",
  });

  useEffect(() => {
    if (user) {
      navigate("/dashboard");
    }
  }, [user, navigate]);

  const handleSignup = async () => {
    try {
      setLoading(true);
      const { email, password, name } = formData;
      const user = await AuthService.signup(email, password, name);

      if (user) {
        setUser(user); // update context
        toast({
          title: "Signup successful",
          description: "Please check your email for verification.",
        });
        navigate("/dashboard");
      }
    } catch (err: any) {
      toast({
        title: "Signup Failed",
        description: err.message,
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

const handleLogin = async () => {
  try {
    setLoading(true);
    const { email, password } = formData;
    
    console.log("Attempting login with:", email); // Debug log
    
    const user = await AuthService.login(email, password);

    if (user) {
      // Ensure profile exists
      try {
        await AuthService.ensureProfile(user.id, user.email || "", user.user_metadata?.full_name || "");
      } catch (profileErr) {
        console.log("Profile check:", profileErr);
      }

      setUser(user);
      toast({
        title: "Welcome Back!",
        description: "You have logged in successfully.",
      });
      navigate("/dashboard");
    }
  } catch (err: any) {
    console.error("Login error details:", err); // More detailed error
    toast({
      title: "Login Failed",
      description: err.message || "Invalid email or password",
      variant: "destructive",
    });
  } finally {
    setLoading(false);
  }
};

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    isLogin ? handleLogin() : handleSignup();
  };

  return (
    <div className="min-h-screen flex">
      {/* LEFT SIDE */}
      <div className="hidden lg:flex lg:w-1/2 xl:w-[55%] gradient-primary relative overflow-hidden">
        <div className="absolute inset-y-0 right-0 w-32">
          <svg
            viewBox="0 0 100 100"
            preserveAspectRatio="none"
            className="h-full w-full"
            fill="hsl(var(--background))"
          >
            <path d="M100 0 C30 0, 30 100, 100 100 L100 0 Z" />
          </svg>
        </div>

        <div className="relative z-10 flex flex-col w-full p-12">
          <div className="flex items-center gap-3 mb-auto">
            <div className="w-10 h-10 rounded-lg gradient-accent flex items-center justify-center">
              <BarChart3 className="w-6 h-6 text-accent-foreground" />
            </div>
            <span className="text-primary-foreground font-display text-xl font-bold">
              DataViz Pro
            </span>
          </div>

          <div className="flex-1 flex items-center justify-center py-8">
            <DataViz />
          </div>

          <div className="mt-auto">
            <h2 className="text-primary-foreground font-display text-3xl font-bold mb-3">
              Transform Data Into
              <br />
              <span className="text-gradient">Actionable Insights</span>
            </h2>
            <p className="text-primary-foreground/70 text-base max-w-md">
              Create stunning visualizations, interactive dashboards, and powerful analytics.
            </p>
          </div>
        </div>
      </div>

      {/* RIGHT SIDE (FORM) */}
      <div className="flex-1 flex flex-col justify-center px-8 sm:px-12 lg:px-16 xl:px-24 bg-background">
        <div className="w-full max-w-md mx-auto">
          <div className="mb-8 animate-fade-in-up">
            <h1 className="text-3xl font-display font-bold text-foreground mb-2">
              {isLogin ? "Welcome Back" : "Create Account"}
            </h1>
            <p className="text-muted-foreground">
              {isLogin ? "Sign in to access your dashboard" : "Start your journey today"}
            </p>
          </div>

          {/* FORM */}
          <form onSubmit={handleSubmit} className="space-y-5 animate-fade-in-up animation-delay-100">
            {!isLogin && (
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <div className="relative">
                  <User className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                  <Input
                    id="name"
                    type="text"
                    placeholder="Enter your name"
                    className="pl-12"
                    value={formData.name}
                    onChange={(e) =>
                      setFormData({ ...formData, name: e.target.value })
                    }
                  />
                </div>
              </div>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  placeholder="Enter your email"
                  className="pl-12"
                  value={formData.email}
                  onChange={(e) =>
                    setFormData({ ...formData, email: e.target.value })
                  }
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Lock className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  id="password"
                  type={showPassword ? "text" : "password"}
                  placeholder="Enter your password"
                  className="pl-12 pr-12"
                  value={formData.password}
                  onChange={(e) =>
                    setFormData({ ...formData, password: e.target.value })
                  }
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-4 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                >
                  {showPassword ? <EyeOff /> : <Eye />}
                </button>
              </div>
            </div>

            <Button type="submit" size="lg" className="w-full mt-6" disabled={loading}>
              {loading ? "Please wait..." : isLogin ? "Sign In" : "Create Account"}
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </form>

          {/* Toggle */}
          <p className="mt-8 text-center text-muted-foreground">
            {isLogin ? "Don't have an account?" : "Already have an account?"}{" "}
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="text-accent font-semibold"
            >
              {isLogin ? "Sign Up" : "Sign In"}
            </button>
          </p>
        </div>
      </div>
    </div>
  );
};

export default Auth;
