import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";

import Auth from "./pages/Auth";
import Index from "./pages/Index";
import Dashboard from "./pages/Dashboard";
import NotFound from "./pages/NotFound";

import ProtectedRoute from "@/services/auth/ProtectedRoute";

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />

        <BrowserRouter>
          <Routes>
            {/* PUBLIC ROUTES */}
            <Route path="/" element={<Index />} />
            <Route path="/auth" element={<Auth />} />

            {/* ALL ROUTES BELOW THIS ARE PROTECTED */}
            <Route element={<ProtectedRoute />}>
              <Route path="/dashboard" element={<Dashboard />} />
              {/* If you add more pages, they are auto-protected */}
              {/* <Route path="/datasets" element={<Datasets />} /> */}
              {/* <Route path="/profile" element={<Profile />} /> */}
            </Route>

            {/* 404 */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>

      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
