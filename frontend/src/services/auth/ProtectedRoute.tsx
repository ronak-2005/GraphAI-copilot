// src/services/auth/ProtectedRoute.tsx
import { Navigate, Outlet } from "react-router-dom";
import { useAuth } from "@/hooks/useAuth";
import { useEffect, useState } from "react";

const ProtectedRoute = () => {
  const { user } = useAuth();
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Give time for auth to load
    const timer = setTimeout(() => {
      setLoading(false);
    }, 100);

    return () => clearTimeout(timer);
  }, [user]);

  // Show loading while checking auth
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // If no user after loading, redirect to auth
  if (!user) {
    return <Navigate to="/auth" replace />;
  }

  // User is authenticated, render protected routes
  return <Outlet />;
};

export default ProtectedRoute;