"""Supabase client initialization module."""

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client

# Locate and load root .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def get_supabase_client(use_service_role: bool = True) -> Client:
    """Creates and returns a Supabase client.

    Args:
        use_service_role: If True, uses the SUPABASE_SERVICE_ROLE_KEY for admin/backend
            operations (Storage management, bypassing RLS). If False, uses
            SUPABASE_ANON_KEY.

    Returns:
        Client: Supabase client instance.
    """
    url = os.getenv("SUPABASE_URL")
    if not url:
        raise ValueError("SUPABASE_URL environment variable is required.")

    if use_service_role:
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not key:
            raise ValueError(
                "SUPABASE_SERVICE_ROLE_KEY environment variable is required for admin client."
            )
    else:
        key = os.getenv("SUPABASE_ANON_KEY")
        if not key:
            raise ValueError(
                "SUPABASE_ANON_KEY environment variable is required."
            )

    return create_client(url, key)
