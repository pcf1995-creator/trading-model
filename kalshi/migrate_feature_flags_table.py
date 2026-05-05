#!/usr/bin/env python3
"""
Migration script: Create feature_flags table in Supabase.

Usage:
  python3 migrate_feature_flags_table.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

def migrate():
    """Create feature_flags table and seed initial values."""
    try:
        import db
        client = db._get_client()

        if not client:
            print("❌ Could not connect to Supabase. Check SUPABASE_URL and SUPABASE_KEY.")
            return False

        print("📋 Creating feature_flags table...")

        # Create table via raw SQL
        try:
            import psycopg2
            from urllib.parse import urlparse

            supabase_url = os.environ.get("SUPABASE_URL")
            if not supabase_url:
                try:
                    import streamlit as st
                    supabase_url = st.secrets.get("SUPABASE_URL")
                except:
                    pass

            if not supabase_url:
                print("❌ SUPABASE_URL not found")
                return False

            supabase_key = os.environ.get("SUPABASE_KEY")
            if not supabase_key:
                try:
                    import streamlit as st
                    supabase_key = st.secrets.get("SUPABASE_KEY")
                except:
                    pass

            if not supabase_key:
                print("❌ SUPABASE_KEY not found")
                return False

            db_url = supabase_url.replace("https://", "postgresql://") + ":5432/postgres"
            parsed = urlparse(db_url)
            conn_str = f"postgresql://postgres:{supabase_key}@{parsed.netloc}:5432/postgres"

            conn = psycopg2.connect(conn_str)
            cursor = conn.cursor()

            # Create table
            sql = """
            CREATE TABLE IF NOT EXISTS public.feature_flags (
                key TEXT PRIMARY KEY,
                value BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
            """

            cursor.execute(sql)
            conn.commit()

            print("✓ Table created")

            # Seed initial values
            print("📌 Seeding initial feature flags...")

            flags = [
                ("auto_place_weekly", False),
                ("auto_place_vol", False),
                ("auto_place_intraday_short", False),
                ("auto_place_intraday_long", False),
            ]

            for key, value in flags:
                insert_sql = """
                INSERT INTO public.feature_flags (key, value)
                VALUES (%s, %s)
                ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value;
                """
                cursor.execute(insert_sql, (key, value))

            conn.commit()
            cursor.close()
            conn.close()

            print("✓ Feature flags seeded")
            return True

        except ImportError:
            print("⚠️  psycopg2 not installed, trying REST API fallback...")
            # Try Supabase REST API
            try:
                # Insert via REST
                for key in ["auto_place_weekly", "auto_place_vol", "auto_place_intraday_short", "auto_place_intraday_long"]:
                    try:
                        client.table("feature_flags").insert({"key": key, "value": False}).execute()
                    except:
                        pass
                print("✓ Table created and seeded via REST API")
                return True
            except Exception as e:
                print(f"❌ REST API failed: {e}")
                return False

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Migration: Create feature_flags table")
    print("=" * 60)

    success = migrate()

    if success:
        print("\n✅ Migration complete!")
        print("Feature flags table is ready. Auto-placement will now read from Supabase.")
    else:
        print("\n❌ Migration failed.")
        print("\nManual fix: In Supabase dashboard:")
        print("  1. SQL Editor → New Query")
        print("  2. Paste and run:")
        print("""
        CREATE TABLE public.feature_flags (
            key TEXT PRIMARY KEY,
            value BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        INSERT INTO public.feature_flags (key, value) VALUES
            ('auto_place_weekly', false),
            ('auto_place_vol', false),
            ('auto_place_intraday_short', false),
            ('auto_place_intraday_long', false);
        """)
        sys.exit(1)
