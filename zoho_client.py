# Minimal Zoho CRM client for the search_zoho_lead tool (mcp_tools.py). Just
# OAuth token exchange (refresh_token grant) and one search call -- not a
# general-purpose SDK. Every other tool in this demo stays a mock; this is
# the one intent (lookup_crm_record) wired to a real backend.

import json
import os
import time
import urllib.parse
import urllib.request

from dotenv import load_dotenv

load_dotenv()

TOKEN_URL = "https://accounts.zoho.com/oauth/v2/token"
SEARCH_URL = "https://www.zohoapis.com/crm/v3/Leads/search"

_access_token = None
_token_expires_at = 0.0


def _get_access_token():
    """Exchange the refresh token for an access token, caching it in memory
    until it's close to expiry."""
    global _access_token, _token_expires_at
    if _access_token and time.time() < _token_expires_at:
        return _access_token

    params = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "client_id": os.environ["ZOHO_CLIENT_ID"],
        "client_secret": os.environ["ZOHO_CLIENT_SECRET"],
        "refresh_token": os.environ["ZOHO_REFRESH_TOKEN"],
    }).encode()

    request = urllib.request.Request(TOKEN_URL, data=params, method="POST")
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())

    _access_token = payload["access_token"]
    _token_expires_at = time.time() + payload.get("expires_in", 3600) - 60
    return _access_token


def search_lead(email):
    """Search Zoho CRM Leads by email. Returns the parsed JSON response."""
    token = _get_access_token()
    url = f"{SEARCH_URL}?{urllib.parse.urlencode({'email': email})}"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Zoho-oauthtoken {token}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())
