// src/services/auth/authService.ts
import { supabase } from "../../integrations/supabase/client";

export const AuthService = {
  // LOGIN
  login: async (email: string, password: string) => {
    const { data, error } = await supabase.auth.signInWithPassword({ 
      email, 
      password 
    });
    
    if (error) {
      console.error("Login error:", error);
      throw new Error(error.message || "Login failed");
    }

    if (!data.user) throw new Error("Login failed: no user returned");
    return data.user;
  },

  // SIGNUP
  signup: async (email: string, password: string, full_name: string = "") => {
    // Step 1: Create the auth user
    const { data, error } = await supabase.auth.signUp({ 
      email, 
      password,
      options: {
        data: {
          full_name: full_name,
        }
      }
    });
    
    if (error) {
      console.error("Signup error:", error);
      throw new Error(error.message || "Signup failed");
    }

    const user = data.user;
    if (!user) throw new Error("Signup failed: no user returned");

    // Step 2: Check if session is established
    const { data: sessionData } = await supabase.auth.getSession();
    
    if (!sessionData.session) {
      // If email confirmation is required, profile will be created on first login
      console.log("Email confirmation required - profile will be created on first login");
      return user;
    }

    // Step 3: Now the session is active, create profile with RLS working properly
    const { error: profileErr } = await supabase
      .from("profiles")
      .insert({
        id: user.id,
        full_name,
        username: email.split('@')[0],
        avatar_url: "",
      });

    if (profileErr) {
      console.error("Profile creation error:", profileErr);
      throw new Error("Failed to create profile: " + profileErr.message);
    }

    return user;
  },

  // LOGOUT
  logout: async () => {
    const { error } = await supabase.auth.signOut();
    if (error) throw error;
  },

  // GET CURRENT AUTH USER
  getUser: async () => {
    const { data } = await supabase.auth.getUser();
    return data.user;
  },

  // GET PROFILE
  getProfile: async (id: string) => {
    const { data, error } = await supabase
      .from("profiles")
      .select("*")
      .eq("id", id)
      .single();
    
    if (error) throw error;
    return data;
  },

  // UPDATE PROFILE
  updateProfile: async (id: string, updates: any) => {
    const { data, error } = await supabase
      .from("profiles")
      .update(updates)
      .eq("id", id)
      .select()
      .single();
    
    if (error) throw error;
    return data;
  },

  // Create profile if it doesn't exist (helper function)
  ensureProfile: async (userId: string, email: string, full_name: string = "") => {
    // Check if profile exists
    const { data: existingProfile } = await supabase
      .from("profiles")
      .select("id")
      .eq("id", userId)
      .single();

    if (existingProfile) {
      return existingProfile;
    }

    // Create profile if it doesn't exist
    const { data, error } = await supabase
      .from("profiles")
      .insert({
        id: userId,
        full_name,
        username: email.split('@')[0],
        avatar_url: "",
      })
      .select()
      .single();

    if (error) throw error;
    return data;
  },
};