# Check MCP connections before (or after) binding catalog tools to them:
# validates connections.json and the environment variables it reads, then
# connects and lists each server's tools. Secrets are never printed.
#
#     python scripts/check_connections.py            # every connection
#     python scripts/check_connections.py hubspot    # just one
#
# Exit code 1 if any checked connection is misconfigured or unreachable.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ciap import config  # noqa: E402
from ciap.integrations import mcp_client  # noqa: E402


def _flags(tool):
    hints = tool["annotations"]
    flags = []
    if hints.get("readOnlyHint"):
        flags.append("read-only")
    if hints.get("destructiveHint"):
        flags.append("DESTRUCTIVE")
    if not tool["allowed"]:
        flags.append("not in allowed_tools")
    return f"  [{', '.join(flags)}]" if flags else ""


def main():
    parser = argparse.ArgumentParser(description="Check MCP connections from connections.json.")
    parser.add_argument("names", nargs="*", help="connection names (default: all)")
    args = parser.parse_args()

    print(f"Connections file: {config.MCP_CONNECTIONS_PATH}")
    problems = mcp_client.check_connections()
    if not problems:
        print("No connections configured.")
        return 1
    names = args.names or sorted(problems)
    failed = False
    try:
        for name in names:
            print(f"\n{name}")
            if name not in problems:
                print("  NOT CONFIGURED")
                failed = True
                continue
            if problems[name]:
                print(f"  CONFIG ERROR: {problems[name]}")
                failed = True
                continue
            spec = mcp_client._connection_spec(name)
            auth = spec.auth.type if spec.auth else "none"
            print(f"  transport={spec.transport} auth={auth} timeout={mcp_client._timeout(spec):g}s")
            try:
                tools = mcp_client.list_tools(name)
            except Exception as exc:
                print(f"  CONNECTION FAILED: {spec.redact(f'{type(exc).__name__}: {exc}')}")
                failed = True
                continue
            print(f"  OK, {len(tools)} tools:")
            for tool in tools:
                print(f"    {tool['name']}{_flags(tool)}")
    finally:
        mcp_client.shutdown()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
