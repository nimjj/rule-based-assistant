# Remote bindings for the shipped Zoho CRM intents (see intents.py). They all
# run the ZohoCRM_searchRecords tool on a Zoho-hosted MCP server (set up in
# Zoho's MCP console, with that tool enabled). For another server, check its
# tools with GET /v1/connections/{name}/tools and change an intent's binding
# with PUT /v1/intents/{name}. The binding format is described above
# mcp_tools.remote_calls.


def _search(module, query_params, fields, fallback=None):
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
LEAD_SEARCH = _search(
    "Leads", {"email": "{email}"}, _LEAD_FIELDS,
    fallback=_search("Contacts", {"email": "{email}"}, _CONTACT_FIELDS),
)
# Someone calling in by phone is more likely an existing customer: Contacts first.
PHONE_SEARCH = _search(
    "Contacts", {"phone": "{phone}"}, _CONTACT_FIELDS,
    fallback=_search("Leads", {"phone": "{phone}"}, _LEAD_FIELDS),
)
ACCOUNT_SEARCH = _search("Accounts", {"word": "{company_name}"}, _ACCOUNT_FIELDS)
DEAL_SEARCH = _search("Deals", {"word": "{deal_name}"}, _DEAL_FIELDS)
PRODUCT_SEARCH = _search("Products", {"word": "{product_name}"}, _PRODUCT_FIELDS)
