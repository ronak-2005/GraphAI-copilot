export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[]

export type Database = {
  // Allows to automatically instantiate createClient with right options
  // instead of createClient<Database, { PostgrestVersion: 'XX' }>(URL, KEY)
  __InternalSupabase: {
    PostgrestVersion: "13.0.5"
  }
  public: {
    Tables: {
      chat_history: {
        Row: {
          context: Json | null
          created_at: string
          dashboard_id: string | null
          dataset_id: string | null
          id: string
          message: string
          message_type: string | null
          response: string
          user_id: string
        }
        Insert: {
          context?: Json | null
          created_at?: string
          dashboard_id?: string | null
          dataset_id?: string | null
          id?: string
          message: string
          message_type?: string | null
          response: string
          user_id: string
        }
        Update: {
          context?: Json | null
          created_at?: string
          dashboard_id?: string | null
          dataset_id?: string | null
          id?: string
          message?: string
          message_type?: string | null
          response?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "chat_history_dashboard_id_fkey"
            columns: ["dashboard_id"]
            isOneToOne: false
            referencedRelation: "dashboards"
            referencedColumns: ["id"]
          },
          {
            foreignKeyName: "chat_history_dataset_id_fkey"
            columns: ["dataset_id"]
            isOneToOne: false
            referencedRelation: "datasets"
            referencedColumns: ["id"]
          },
        ]
      }
      dashboards: {
        Row: {
          created_at: string
          dataset_id: string
          description: string | null
          id: string
          is_public: boolean | null
          layout_config: Json | null
          name: string
          public_share_id: string | null
          theme: string | null
          updated_at: string
          user_id: string
        }
        Insert: {
          created_at?: string
          dataset_id: string
          description?: string | null
          id?: string
          is_public?: boolean | null
          layout_config?: Json | null
          name: string
          public_share_id?: string | null
          theme?: string | null
          updated_at?: string
          user_id: string
        }
        Update: {
          created_at?: string
          dataset_id?: string
          description?: string | null
          id?: string
          is_public?: boolean | null
          layout_config?: Json | null
          name?: string
          public_share_id?: string | null
          theme?: string | null
          updated_at?: string
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "dashboards_dataset_id_fkey"
            columns: ["dataset_id"]
            isOneToOne: false
            referencedRelation: "datasets"
            referencedColumns: ["id"]
          },
        ]
      }
      data: {
        Row: {
          created_at: string
          id: number
          name: string
        }
        Insert: {
          created_at?: string
          id?: number
          name: string
        }
        Update: {
          created_at?: string
          id?: number
          name?: string
        }
        Relationships: []
      }
      datasets: {
        Row: {
          clickhouse_table_name: string | null
          column_count: number | null
          columns_metadata: Json | null
          created_at: string
          description: string | null
          error_message: string | null
          file_key: string | null
          file_size: number
          file_url: string | null
          id: string
          name: string
          row_count: number | null
          source_type: string
          status: string
          storage_location: string
          updated_at: string
          user_id: string
        }
        Insert: {
          clickhouse_table_name?: string | null
          column_count?: number | null
          columns_metadata?: Json | null
          created_at?: string
          description?: string | null
          error_message?: string | null
          file_key?: string | null
          file_size?: number
          file_url?: string | null
          id?: string
          name: string
          row_count?: number | null
          source_type: string
          status?: string
          storage_location: string
          updated_at?: string
          user_id: string
        }
        Update: {
          clickhouse_table_name?: string | null
          column_count?: number | null
          columns_metadata?: Json | null
          created_at?: string
          description?: string | null
          error_message?: string | null
          file_key?: string | null
          file_size?: number
          file_url?: string | null
          id?: string
          name?: string
          row_count?: number | null
          source_type?: string
          status?: string
          storage_location?: string
          updated_at?: string
          user_id?: string
        }
        Relationships: []
      }
      ml_insights: {
        Row: {
          confidence_score: number
          created_at: string
          data: Json
          dataset_id: string
          description: string
          id: string
          insight_type: string
          is_dismissed: boolean | null
          title: string
          visualization_config: Json | null
        }
        Insert: {
          confidence_score: number
          created_at?: string
          data: Json
          dataset_id: string
          description: string
          id?: string
          insight_type: string
          is_dismissed?: boolean | null
          title: string
          visualization_config?: Json | null
        }
        Update: {
          confidence_score?: number
          created_at?: string
          data?: Json
          dataset_id?: string
          description?: string
          id?: string
          insight_type?: string
          is_dismissed?: boolean | null
          title?: string
          visualization_config?: Json | null
        }
        Relationships: [
          {
            foreignKeyName: "ml_insights_dataset_id_fkey"
            columns: ["dataset_id"]
            isOneToOne: false
            referencedRelation: "datasets"
            referencedColumns: ["id"]
          },
        ]
      }
      processing_jobs: {
        Row: {
          completed_at: string | null
          created_at: string
          current_step: string | null
          dataset_id: string
          error_message: string | null
          file_key: string
          id: string
          job_type: string
          progress: number | null
          rows_processed: number | null
          started_at: string | null
          status: string
          total_rows: number | null
          user_id: string
        }
        Insert: {
          completed_at?: string | null
          created_at?: string
          current_step?: string | null
          dataset_id: string
          error_message?: string | null
          file_key: string
          id?: string
          job_type: string
          progress?: number | null
          rows_processed?: number | null
          started_at?: string | null
          status?: string
          total_rows?: number | null
          user_id: string
        }
        Update: {
          completed_at?: string | null
          created_at?: string
          current_step?: string | null
          dataset_id?: string
          error_message?: string | null
          file_key?: string
          id?: string
          job_type?: string
          progress?: number | null
          rows_processed?: number | null
          started_at?: string | null
          status?: string
          total_rows?: number | null
          user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "processing_jobs_dataset_id_fkey"
            columns: ["dataset_id"]
            isOneToOne: false
            referencedRelation: "datasets"
            referencedColumns: ["id"]
          },
        ]
      }
      profiles: {
        Row: {
          avatar_url: string | null
          created_at: string | null
          full_name: string | null
          id: string
          username: string | null
        }
        Insert: {
          avatar_url?: string | null
          created_at?: string | null
          full_name?: string | null
          id: string
          username?: string | null
        }
        Update: {
          avatar_url?: string | null
          created_at?: string | null
          full_name?: string | null
          id?: string
          username?: string | null
        }
        Relationships: []
      }
      query_cache_metadata: {
        Row: {
          created_at: string
          dataset_id: string
          expires_at: string
          hit_count: number | null
          id: string
          query_hash: string
          query_params: Json
          redis_key: string
          result_size_bytes: number
        }
        Insert: {
          created_at?: string
          dataset_id: string
          expires_at: string
          hit_count?: number | null
          id?: string
          query_hash: string
          query_params: Json
          redis_key: string
          result_size_bytes: number
        }
        Update: {
          created_at?: string
          dataset_id?: string
          expires_at?: string
          hit_count?: number | null
          id?: string
          query_hash?: string
          query_params?: Json
          redis_key?: string
          result_size_bytes?: number
        }
        Relationships: [
          {
            foreignKeyName: "query_cache_metadata_dataset_id_fkey"
            columns: ["dataset_id"]
            isOneToOne: false
            referencedRelation: "datasets"
            referencedColumns: ["id"]
          },
        ]
      }
      shared_dashboards: {
        Row: {
          created_at: string
          dashboard_id: string
          id: string
          permission_level: string
          shared_by_user_id: string
          shared_with_user_id: string
        }
        Insert: {
          created_at?: string
          dashboard_id: string
          id?: string
          permission_level: string
          shared_by_user_id: string
          shared_with_user_id: string
        }
        Update: {
          created_at?: string
          dashboard_id?: string
          id?: string
          permission_level?: string
          shared_by_user_id?: string
          shared_with_user_id?: string
        }
        Relationships: [
          {
            foreignKeyName: "shared_dashboards_dashboard_id_fkey"
            columns: ["dashboard_id"]
            isOneToOne: false
            referencedRelation: "dashboards"
            referencedColumns: ["id"]
          },
        ]
      }
      visualizations: {
        Row: {
          chart_config: Json | null
          chart_type: string
          created_at: string
          dashboard_id: string
          data_config: Json
          id: string
          position: Json
          title: string
          updated_at: string
        }
        Insert: {
          chart_config?: Json | null
          chart_type: string
          created_at?: string
          dashboard_id: string
          data_config: Json
          id?: string
          position?: Json
          title: string
          updated_at?: string
        }
        Update: {
          chart_config?: Json | null
          chart_type?: string
          created_at?: string
          dashboard_id?: string
          data_config?: Json
          id?: string
          position?: Json
          title?: string
          updated_at?: string
        }
        Relationships: [
          {
            foreignKeyName: "visualizations_dashboard_id_fkey"
            columns: ["dashboard_id"]
            isOneToOne: false
            referencedRelation: "dashboards"
            referencedColumns: ["id"]
          },
        ]
      }
    }
    Views: {
      [_ in never]: never
    }
    Functions: {
      [_ in never]: never
    }
    Enums: {
      [_ in never]: never
    }
    CompositeTypes: {
      [_ in never]: never
    }
  }
}

type DatabaseWithoutInternals = Omit<Database, "__InternalSupabase">

type DefaultSchema = DatabaseWithoutInternals[Extract<keyof Database, "public">]

export type Tables<
  DefaultSchemaTableNameOrOptions extends
    | keyof (DefaultSchema["Tables"] & DefaultSchema["Views"])
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
        DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? (DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"] &
      DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Views"])[TableName] extends {
      Row: infer R
    }
    ? R
    : never
  : DefaultSchemaTableNameOrOptions extends keyof (DefaultSchema["Tables"] &
        DefaultSchema["Views"])
    ? (DefaultSchema["Tables"] &
        DefaultSchema["Views"])[DefaultSchemaTableNameOrOptions] extends {
        Row: infer R
      }
      ? R
      : never
    : never

export type TablesInsert<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Insert: infer I
    }
    ? I
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Insert: infer I
      }
      ? I
      : never
    : never

export type TablesUpdate<
  DefaultSchemaTableNameOrOptions extends
    | keyof DefaultSchema["Tables"]
    | { schema: keyof DatabaseWithoutInternals },
  TableName extends DefaultSchemaTableNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"]
    : never = never,
> = DefaultSchemaTableNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaTableNameOrOptions["schema"]]["Tables"][TableName] extends {
      Update: infer U
    }
    ? U
    : never
  : DefaultSchemaTableNameOrOptions extends keyof DefaultSchema["Tables"]
    ? DefaultSchema["Tables"][DefaultSchemaTableNameOrOptions] extends {
        Update: infer U
      }
      ? U
      : never
    : never

export type Enums<
  DefaultSchemaEnumNameOrOptions extends
    | keyof DefaultSchema["Enums"]
    | { schema: keyof DatabaseWithoutInternals },
  EnumName extends DefaultSchemaEnumNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"]
    : never = never,
> = DefaultSchemaEnumNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[DefaultSchemaEnumNameOrOptions["schema"]]["Enums"][EnumName]
  : DefaultSchemaEnumNameOrOptions extends keyof DefaultSchema["Enums"]
    ? DefaultSchema["Enums"][DefaultSchemaEnumNameOrOptions]
    : never

export type CompositeTypes<
  PublicCompositeTypeNameOrOptions extends
    | keyof DefaultSchema["CompositeTypes"]
    | { schema: keyof DatabaseWithoutInternals },
  CompositeTypeName extends PublicCompositeTypeNameOrOptions extends {
    schema: keyof DatabaseWithoutInternals
  }
    ? keyof DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"]
    : never = never,
> = PublicCompositeTypeNameOrOptions extends {
  schema: keyof DatabaseWithoutInternals
}
  ? DatabaseWithoutInternals[PublicCompositeTypeNameOrOptions["schema"]]["CompositeTypes"][CompositeTypeName]
  : PublicCompositeTypeNameOrOptions extends keyof DefaultSchema["CompositeTypes"]
    ? DefaultSchema["CompositeTypes"][PublicCompositeTypeNameOrOptions]
    : never

export const Constants = {
  public: {
    Enums: {},
  },
} as const
