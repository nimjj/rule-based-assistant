# Sign in to an MCP connection that uses the MCP standard sign-in
# ("auth": {"type": "mcp_oauth"} in connections.json). Opens the server's
# sign-in page in your browser; once you approve, the sign-in is saved to
# data/oauth/<name>.json and the server keeps it refreshed. Run it again
# whenever a connection reports it needs signing in.
#
#     python scripts/connect.py hubspot
#
# The browser must run on this machine (the redirect comes back to
# localhost). To sign in for a remote server, run this locally and copy
# data/oauth/<name>.json to the server's data/oauth/.

import argparse
import os
import sys
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ciap.integrations import mcp_client  # noqa: E402


def show_sign_in_page(url):
    print(f"Opening the sign-in page in your browser. If it doesn't open, go to:\n\n{url}\n")
    webbrowser.open(url)
    print("Waiting for you to approve...")


def main():
    parser = argparse.ArgumentParser(description="Sign in to an mcp_oauth connection.")
    parser.add_argument("name", help="connection name in connections.json")
    args = parser.parse_args()
    try:
        tools = mcp_client.sign_in(args.name, show_sign_in_page)
    except (mcp_client.ConnectionNotConfigured, mcp_client.ToolCallFailed) as exc:
        print(f"Sign-in failed: {exc}")
        return 1
    except Exception as exc:
        print(f"Sign-in failed: {mcp_client._describe(exc)}")
        return 1
    print(f"\nSigned in to {args.name}: {len(tools)} tools available.")
    print(f"Saved to {mcp_client._token_path(args.name)}. A running API server uses it from its next call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
