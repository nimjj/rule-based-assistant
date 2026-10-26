# Step 6: a mock MCP tool catalog -- the backend actions a fired intent frame
# actually maps to. Shaped like real MCP tool definitions (name / description
# / inputSchema) so the dialogue manager can hand off a completed frame as a
# tool call the way an MCP-aware LLM client would, instead of every call site
# re-deriving "which action, which arguments" by hand. Most tools are mocks;
# a tool with a `remote` binding really executes, as a call to a tool on a
# remote MCP server (see the binding notes above dispatch_tool_call, and mcp_client.py).

import re

from ciap.integrations import mcp_client

# The shipped live bindings all run the ZohoCRM_searchRecords tool on a
# Zoho-hosted MCP server (set up in Zoho's MCP console, with that tool
# enabled). For another server, check its tools with
# GET /v1/connections/{name}/tools and change a binding with
# PUT /v1/tools/{name}.


def _zoho_search(module, query_params, fields, fallback=None):
    # `fields` keeps the response to what's worth briefing the agent on: a
    # full Zoho record runs to 30-65 fields, and every one slows the answer.
    binding = {
        "connection": "zoho",
        "tool": "ZohoCRM_searchRecords",
        "arguments": {"path_variables": {"module": module}, "query_params": {**query_params, "fields": fields}},
    }
    if fallback:
        binding["fallback"] = fallback
    return binding


_LEAD_FIELDS = ("Full_Name,Email,Company,Lead_Status,Lead_Source,Owner,Phone,Mobile,"
                "Designation,Industry,City,Country,Last_Activity_Time")
_CONTACT_FIELDS = ("Full_Name,Email,Account_Name,Owner,Phone,Mobile,Title,Department,"
                   "Mailing_City,Mailing_Country,Last_Activity_Time,Lead_Source")
_ACCOUNT_FIELDS = ("Account_Name,Owner,Phone,Website,Industry,Account_Type,Billing_City,"
                   "Billing_Country,Employees,Annual_Revenue")
_DEAL_FIELDS = "Deal_Name,Stage,Amount,Closing_Date,Owner,Account_Name,Contact_Name,Probability,Next_Step,Type"
_PRODUCT_FIELDS = "Product_Name,Product_Code,Unit_Price,Qty_in_Stock,Product_Active,Description"

# A caller is a Lead until converted, then a Contact -- and doesn't know
# which. So: Leads first, Contacts when no lead matches.
ZOHO_LEAD_SEARCH_REMOTE = _zoho_search(
    "Leads", {"email": "{email}"}, _LEAD_FIELDS,
    fallback=_zoho_search("Contacts", {"email": "{email}"}, _CONTACT_FIELDS),
)
# Someone calling in by phone is more likely an existing customer: Contacts first.
ZOHO_PHONE_SEARCH_REMOTE = _zoho_search(
    "Contacts", {"phone": "{phone}"}, _CONTACT_FIELDS,
    fallback=_zoho_search("Leads", {"phone": "{phone}"}, _LEAD_FIELDS),
)
ZOHO_ACCOUNT_SEARCH_REMOTE = _zoho_search("Accounts", {"word": "{company_name}"}, _ACCOUNT_FIELDS)
ZOHO_DEAL_SEARCH_REMOTE = _zoho_search("Deals", {"word": "{deal_name}"}, _DEAL_FIELDS)
ZOHO_PRODUCT_SEARCH_REMOTE = _zoho_search("Products", {"word": "{product_name}"}, _PRODUCT_FIELDS)

TOOLS = {
    "get_order_status": {
        "label": "Get Order Status",
        "description": "Look up the shipping/tracking status of a customer's order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference, e.g. ORD-48213."},
            },
            "required": ["order_id"],
        },
    },
    "cancel_order": {
        "label": "Cancel Order",
        "description": "Cancel an order (or a recurring/subscription order) before it ships.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference to cancel."},
            },
            "required": ["order_id"],
        },
    },
    "update_order": {
        "label": "Update Order",
        "description": "Apply a change to an existing order -- shipping address, item, size/color, or merging shipments.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference to modify."},
            },
            "required": ["order_id"],
        },
    },
    "reschedule_delivery": {
        "label": "Reschedule Delivery",
        "description": "Move an order's delivery to a new date.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference."},
                "new_date": {"type": "string", "description": "ISO-8601 date delivery should move to."},
            },
            "required": ["order_id", "new_date"],
        },
    },
    "report_delivery_issue": {
        "label": "Report Delivery Issue",
        "description": "File a delivery problem report (damaged, missing, wrong item, wrong address) against an order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference the problem relates to."},
            },
            "required": ["order_id"],
        },
    },
    "troubleshoot_payment": {
        "label": "Troubleshoot Payment",
        "description": "Diagnose a declined or failed payment attempt at checkout.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_payment_info": {
        "label": "Get Payment Info",
        "description": "Look up accepted payment methods and billing timing/policy.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "file_billing_dispute": {
        "label": "File Billing Dispute",
        "description": "Open a dispute/refund request for an incorrect or duplicate charge on a specific order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference being disputed."},
            },
            "required": ["order_id"],
        },
    },
    "send_invoice": {
        "label": "Send Invoice",
        "description": "Email a receipt/itemized invoice for a specific order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference the invoice is for."},
            },
            "required": ["order_id"],
        },
    },
    "get_product_availability": {
        "label": "Check Product Availability",
        "description": "Check stock/size/color availability for a named product. Qty_in_Stock is how many are in stock.",
        "remote": ZOHO_PRODUCT_SEARCH_REMOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Product name as it appears in the catalog."},
            },
            "required": ["product_name"],
        },
    },
    "get_product_details": {
        "label": "Get Product Details",
        "description": "Look up specs, materials, warranty, reviews, or sample availability for a named product.",
        "remote": ZOHO_PRODUCT_SEARCH_REMOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string", "description": "Product name as it appears in the catalog."},
            },
            "required": ["product_name"],
        },
    },
    "get_return_policy": {
        "label": "Get Return Policy",
        "description": "Look up general return/exchange policy (window, return shipping, sale items, packaging).",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "start_return": {
        "label": "Start Return",
        "description": "Begin a return for an item on a specific order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference to return from."},
            },
            "required": ["order_id"],
        },
    },
    "get_refund_status": {
        "label": "Get Refund Status",
        "description": "Check the processing status of a refund for a specific order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference the refund is for."},
            },
            "required": ["order_id"],
        },
    },
    "start_exchange": {
        "label": "Start Exchange",
        "description": "Begin an exchange (different size/product) for an item on a specific order.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order reference to exchange from."},
            },
            "required": ["order_id"],
        },
    },
    "get_current_promotions": {
        "label": "Get Current Promotions",
        "description": "List discount codes/promotions currently active.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "validate_discount_code": {
        "label": "Validate Discount Code",
        "description": "Check why a discount code isn't applying, or whether it can still be applied/stacked.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "discount_code": {"type": "string", "description": "The promo code in question, e.g. SAVE20."},
            },
            "required": ["discount_code"],
        },
    },
    "get_loyalty_info": {
        "label": "Get Loyalty Info",
        "description": "Look up loyalty/rewards program details or a customer's points balance.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "submit_price_match_request": {
        "label": "Submit Price Match",
        "description": "Submit a price-match request against a competitor's listing.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "open_account_request": {
        "label": "Open Account Request",
        "description": "Handle an account-management request: password reset, email change, deletion, order history, verification email, saved addresses.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "report_technical_issue": {
        "label": "Report Technical Issue",
        "description": "File a website/technical bug report (checkout, cart, form fields, missing emails).",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "get_gift_card_info": {
        "label": "Get Gift Card Info",
        "description": "Handle gift-card purchase, balance check, gift-wrapping, or expiry questions.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "transfer_to_agent": {
        "label": "Transfer to Agent",
        "description": "Transfer the conversation to a human agent.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "file_complaint": {
        "label": "File Complaint",
        "description": "Log a customer complaint/escalation, optionally tied to an order.",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    "search_zoho_lead": {
        "label": "Zoho CRM Customer Lookup",
        "description": (
            "Search Zoho CRM for the customer by email: leads first, then contacts (existing "
            "customers). A record's Owner is the account rep assigned to it."
        ),
        # Really executes -- see dispatch_tool_call.
        "remote": ZOHO_LEAD_SEARCH_REMOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email to search for in Zoho CRM."},
            },
            "required": ["email"],
        },
    },
    "search_zoho_by_phone": {
        "label": "Zoho CRM Phone Lookup",
        "description": (
            "Search Zoho CRM for the caller by phone number: contacts (existing customers) first, "
            "then leads. A record's Owner is the account rep assigned to it."
        ),
        "remote": ZOHO_PHONE_SEARCH_REMOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "phone": {"type": "string", "description": "Caller's phone number, in any format."},
            },
            "required": ["phone"],
        },
    },
    "search_zoho_account": {
        "label": "Zoho CRM Account Lookup",
        "description": (
            "Search Zoho CRM accounts (companies) by name. An account's Owner is the account rep "
            "assigned to it."
        ),
        "remote": ZOHO_ACCOUNT_SEARCH_REMOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "company_name": {"type": "string", "description": "The caller's company name."},
            },
            "required": ["company_name"],
        },
    },
    "get_zoho_deal": {
        "label": "Zoho CRM Deal Status",
        "description": (
            "Look up a sales deal in Zoho CRM by name: its stage, amount, expected closing date "
            "and next step. A deal's Owner is the sales rep handling it."
        ),
        "remote": ZOHO_DEAL_SEARCH_REMOTE,
        "inputSchema": {
            "type": "object",
            "properties": {
                "deal_name": {"type": "string", "description": "The deal's name, as the caller says it."},
            },
            "required": ["deal_name"],
        },
    },
}

# Which tool a fired intent frame maps to. One tool per intent for this MVP;
# a real catalog could map many-to-one (e.g. several intents resolving to a
# generic `open_ticket` tool with a `category` argument).
INTENT_TO_TOOL = {
    "track_order": "get_order_status",
    "cancel_order": "cancel_order",
    "modify_order": "update_order",
    "change_delivery_date": "reschedule_delivery",
    "report_delivery_problem": "report_delivery_issue",
    "payment_declined": "troubleshoot_payment",
    "payment_info": "get_payment_info",
    "billing_dispute": "file_billing_dispute",
    "request_invoice": "send_invoice",
    "check_product_availability": "get_product_availability",
    "product_information": "get_product_details",
    "return_policy_info": "get_return_policy",
    "initiate_return": "start_return",
    "check_refund_status": "get_refund_status",
    "exchange_item": "start_exchange",
    "check_current_promotions": "get_current_promotions",
    "promo_code_issue": "validate_discount_code",
    "loyalty_program_info": "get_loyalty_info",
    "price_match_request": "submit_price_match_request",
    "account_management": "open_account_request",
    "website_technical_issue": "report_technical_issue",
    "gift_card_inquiry": "get_gift_card_info",
    "request_human_agent": "transfer_to_agent",
    "file_complaint": "file_complaint",
    "lookup_crm_record": "search_zoho_lead",
    "lookup_by_phone": "search_zoho_by_phone",
    "lookup_company_account": "search_zoho_account",
    "check_deal_status": "get_zoho_deal",
}


def build_tool_call(intent, slots):
    """Resolve a fired frame's intent + extracted slots to an MCP tool call:
    which tool to invoke, and which arguments to pass, per that tool's
    inputSchema. Returns None if the intent has no mapped tool.
    """
    tool_name = INTENT_TO_TOOL.get(intent)
    if tool_name is None:
        return None
    schema = TOOLS[tool_name]["inputSchema"]
    arguments = {}
    for prop_name in schema["properties"]:
        value = slots.get(prop_name)
        if value is None:
            continue
        # new_date is stored as {"text": ..., "date": ...}; the tool call
        # only wants the resolved ISO date.
        arguments[prop_name] = value["date"] if isinstance(value, dict) and "date" in value else value
    return {"tool": tool_name, "arguments": arguments}


# A tool's `remote` binding says how it executes:
#   {"connection": "zoho",              -- a name from config.MCP_CONNECTIONS_PATH
#    "tool": "ZohoCRM_searchRecords",   -- the tool's name on that MCP server
#    "arguments": {"path_variables": {"module": "Leads"},   -- what to send it; optional
#                  "query_params": {"email": "{email}"}},
#    "fallback": {...another binding...}}  -- tried if this one finds nothing; optional
# `{name}` in an argument string is replaced by this tool's argument of that
# name (i.e. the slot); an argument that is exactly "{name}" passes the value
# through as-is; anything else is sent unchanged. Without "arguments" this
# tool's own arguments are sent as they are. Tools without a binding are mocks.
_PLACEHOLDER = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def placeholders(template):
    """Every {name} used anywhere in a binding's arguments."""
    if isinstance(template, str):
        return set(_PLACEHOLDER.findall(template))
    if isinstance(template, dict):
        return set().union(*(placeholders(v) for v in template.values())) if template else set()
    if isinstance(template, list):
        return set().union(*(placeholders(v) for v in template)) if template else set()
    return set()


def render_arguments(template, arguments):
    if isinstance(template, str):
        whole = _PLACEHOLDER.fullmatch(template)
        if whole:
            return arguments.get(whole.group(1))
        return _PLACEHOLDER.sub(lambda m: str(arguments.get(m.group(1), "")), template)
    if isinstance(template, dict):
        return {k: render_arguments(v, arguments) for k, v in template.items()}
    if isinstance(template, list):
        return [render_arguments(v, arguments) for v in template]
    return template


def remote_calls(tool_call):
    """The (connection, remote tool, arguments) a tool call executes as --
    its binding, then each fallback in turn -- or [] for a mock tool."""
    calls = []
    remote = TOOLS.get(tool_call["tool"], {}).get("remote")
    while remote is not None:
        template = remote.get("arguments")
        arguments = tool_call["arguments"] if template is None else render_arguments(template, tool_call["arguments"])
        calls.append((remote["connection"], remote["tool"], arguments))
        remote = remote.get("fallback")
    return calls


def executes_live(tool_name):
    return TOOLS.get(tool_name, {}).get("remote") is not None


NO_RESULTS_MESSAGE = "The tool ran but found no results."


def _is_empty(value):
    # Nothing came back: no content at all, or only empty containers, e.g. {},
    # {"data": []}, {"text": ""}. A result with any real value isn't empty.
    if value is None or value == "":
        return True
    if isinstance(value, dict):
        return all(_is_empty(v) for v in value.values())
    if isinstance(value, list):
        return all(_is_empty(v) for v in value)
    return False


def tool_result(status, data=None, message=None):
    """The fixed shape of every executed tool's frame.tool_result:
      {"status": "success",    "data": {...server response...}, "message": None}
      {"status": "no_results", "data": None, "message": NO_RESULTS_MESSAGE}
      {"status": "error",      "data": None, "message": "<ExceptionType>: <why>"}"""
    return {"status": status, "data": data, "message": message}


def dispatch_tool_call(tool_call):
    """Execute a fired frame's tool call on its remote MCP tool -- trying each
    fallback in turn while nothing is found -- and return its tool_result (see
    tool_result above). Returns None for mock tools (no remote binding), which
    the demo just displays.
    """
    if tool_call is None:
        return None
    calls = remote_calls(tool_call)
    if not calls:
        return None
    # The binding, then its fallbacks while each finds nothing.
    for connection, remote_tool, arguments in calls:
        try:
            result = mcp_client.call_tool(connection, remote_tool, arguments)
        except Exception as exc:
            # An unconfigured connection, a network/auth failure, a timeout or an
            # error from the tool shouldn't take down the whole conversation --
            # surface it as a tool error instead.
            return tool_result("error", message=f"{type(exc).__name__}: {exc}")
        if not _is_empty(result):
            return tool_result("success", data=result)
    return tool_result("no_results", message=NO_RESULTS_MESSAGE)
