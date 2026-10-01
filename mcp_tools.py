# Step 6: a mock MCP tool catalog -- the backend actions a fired intent frame
# actually maps to. Shaped like real MCP tool definitions (name / description
# / inputSchema) so the dialogue manager can hand off a completed frame as a
# tool call the way an MCP-aware LLM client would, instead of every call site
# re-deriving "which action, which arguments" by hand. These tools are mocks
# -- nothing here calls a real backend.

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
        "description": "Check stock/size/color availability for a named product.",
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
        "label": "Search Zoho CRM Lead",
        "description": "Search Zoho CRM for a lead matching a customer's email address.",
        # Really executes -- see BACKENDS / dispatch_tool_call.
        "backend": "zoho_lead_search",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email to search for in Zoho CRM."},
            },
            "required": ["email"],
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


def _zoho_lead_search(arguments):
    import zoho_client
    return zoho_client.search_lead(**arguments)


# A tool's "backend" names the real call it executes. Keyed by backend, not
# tool name, so a live tool keeps working when it's renamed. Tools without
# one (every tool but search_zoho_lead) are mocks.
BACKENDS = {"zoho_lead_search": _zoho_lead_search}


def executes_live(tool_name):
    return TOOLS.get(tool_name, {}).get("backend") in BACKENDS


def dispatch_tool_call(tool_call):
    """Execute a fired frame's tool call against its tool's real backend.
    Returns None for mock tools (no backend), which the demo just displays.
    """
    if tool_call is None:
        return None
    backend = BACKENDS.get(TOOLS.get(tool_call["tool"], {}).get("backend"))
    if backend is None:
        return None
    try:
        return backend(tool_call["arguments"])
    except Exception as exc:
        # Network/auth/response-shape failures on this one real backend call
        # shouldn't take down the whole conversation -- surface them as a
        # tool error instead.
        return {"error": f"{type(exc).__name__}: {exc}"}
