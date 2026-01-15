// src/context/AuthContext.tsx
import { createContext, useContext, useEffect, useState } from "react";
import { supabase } from "../integrations/supabase/client";

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  // Run once - check if session exists
  useEffect(() => {
    const loadUser = async () => {
      const { data } = await supabase.auth.getUser();
      setUser(data.user ?? null);
      setLoading(false);
    };

    loadUser();

    // Listen for login/logout events
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
    });

    return () => listener.subscription.unsubscribe();
  }, []);

  return (
    <AuthContext.Provider value={{ user, loading }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
