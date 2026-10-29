# Intent catalogue: one entry per capability -- what a caller says to ask for
# it (exemplars), what has to be collected first (required_slots), and the
# tool call it fires once they are.
#
# Embedding retrieval compares an incoming utterance against these exemplars
# (not against the intent names) and returns the intent whose exemplars are
# the closest match. Add more exemplar phrasings to improve recall.
#
# Each intent is also its own tool: when its frame fires, mcp_tools builds a
# call named after the intent, with its required slots as the arguments.
# `description` says what the call does (the responder briefs the model with
# it). An intent with a `remote` binding really executes, on a remote MCP
# server (format above mcp_tools.remote_calls); the rest are mocks.
#
# Domain: online sales (general e-commerce retailer). Derived from 74
# customer-support FAQs clustered into 24 intents -- see FAQ_INTENTS.md for
# the full FAQ list and the reasoning behind each cluster/split.

from ciap.integrations import zoho


# TODO - stage5: Learning mechansim via LLM at the end of day, Human-in-the-loop
INTENTS = {
    "track_order": {
        "label": "Track Order",
        "description": "Look up the shipping/tracking status of a customer's order.",
        "exemplars": [
            "Where is my order",
            "Can you check the status of my order",
            "I want to track my package",
            "Why hasn't my order shipped yet",
            "What's the estimated delivery date for my order",
            "My tracking number isn't updating",
            "It says out for delivery but I haven't gotten it yet",
        ],
        "required_slots": ["order_id"],
    },
    "cancel_order": {
        "label": "Cancel Order",
        "description": "Cancel an order (or a recurring/subscription order) before it ships.",
        "exemplars": [
            "Can I cancel my order",
            "I'd like to cancel this order",
            "Please cancel my order",
            "I want to stop my subscription order",
            "Can you cancel my recurring order",
        ],
        "required_slots": ["order_id"],
    },
    "modify_order": {
        "label": "Modify Order",
        "description": (
            "Apply a change to an existing order -- shipping address, item, size/color, or merging "
            "shipments."
        ),
        "exemplars": [
            "Can I change my order after placing it",
            "I need to update the shipping address on my order",
            "Can I add another item to the order I just placed",
            "Can I change the size or color of an item I ordered",
            "Can you combine my two orders into one shipment",
        ],
        "required_slots": ["order_id"],
    },
    "change_delivery_date": {
        "label": "Change Delivery Date",
        "description": "Move an order's delivery to a new date.",
        "exemplars": [
            "Can I change my delivery date",
            "Can you reschedule my delivery to next week",
            "I need my package delivered on a different day",
            "Can we move the delivery date",
        ],
        "required_slots": ["order_id", "new_date"],
    },
    "report_delivery_problem": {
        "label": "Delivery Problem",
        "description": (
            "File a delivery problem report (damaged, missing, wrong item, wrong address) against an "
            "order."
        ),
        "exemplars": [
            "My package arrived damaged",
            "It says delivered but I never got it",
            "I received the wrong item",
            "Part of my order is missing",
            "My package went to the wrong address",
        ],
        "required_slots": ["order_id"],
    },
    "payment_declined": {
        "label": "Payment Declined",
        "description": "Diagnose a declined or failed payment attempt at checkout.",
        "exemplars": [
            "My card was declined at checkout",
            "Why didn't my payment go through",
            "I can't get my card to work on your site",
            "Checkout keeps rejecting my payment",
        ],
        "required_slots": [],
    },
    "payment_info": {
        "label": "Payment Information",
        "description": "Look up accepted payment methods and billing timing/policy.",
        "exemplars": [
            "What payment methods do you accept",
            "Do you accept PayPal or Apple Pay",
            "Can I use a gift card to pay",
            "Can I split payment across two cards",
            "When will my card actually be charged",
        ],
        "required_slots": [],
    },
    "billing_dispute": {
        "label": "Billing Dispute",
        "description": (
            "Open a dispute/refund request for an incorrect or duplicate charge on a specific order."
        ),
        "exemplars": [
            "I was charged twice for one order",
            "The charge on my statement doesn't match my order total",
            "I think I've been overcharged",
            "There's an incorrect charge on my account",
        ],
        "required_slots": ["order_id"],
    },
    "request_invoice": {
        "label": "Request Invoice",
        "description": "Email a receipt/itemized invoice for a specific order.",
        "exemplars": [
            "Can I get a copy of my receipt",
            "I need an invoice for my order",
            "Can you send me an itemized invoice",
            "Where can I download my receipt",
            "Can I get a breakdown of the charges on my order",
        ],
        "required_slots": ["order_id"],
    },
    "check_product_availability": {
        "label": "Product Availability",
        "description": (
            "Check stock/size/color availability for a named product. Qty_in_Stock is how many are in"
            " stock."
        ),
        "remote": zoho.PRODUCT_SEARCH,
        "exemplars": [
            "Is the Aurora Desk Lamp in stock",
            "When will the Marlow Sofa be back in stock",
            "Do you have the Nimbus Jacket in a different size",
            "Is the Rowan Backpack available in black",
            "When is the Nimbus Jacket coming back in stock",
            "Do you have any of the Marlow Sofa left",
        ],
        "required_slots": ["product_name"],
    },
    "product_information": {
        "label": "Product Information",
        "description": (
            "Look up specs, materials, warranty, reviews, or sample availability for a named product."
        ),
        "remote": zoho.PRODUCT_SEARCH,
        "exemplars": [
            "What are the dimensions of the Marlow Sofa",
            "How big is the Aurora Desk Lamp",
            "What material is the Nimbus Jacket made of",
            "Does the Aurora Desk Lamp come with a warranty",
            "Are there reviews for the Rowan Backpack",
            "Can I get a sample of the Cedar Candle before I buy",
        ],
        "required_slots": ["product_name"],
    },
    "return_policy_info": {
        "label": "Return Policy",
        "description": (
            "Look up general return/exchange policy (window, return shipping, sale items, packaging)."
        ),
        "exemplars": [
            "What's your return policy",
            "How long do I have to return something",
            "Do I have to pay for return shipping",
            "Can I return a clearance item",
            "Can I return something without the original packaging",
        ],
        "required_slots": [],
    },
    "initiate_return": {
        "label": "Start a Return",
        "description": "Begin a return for an item on a specific order.",
        "exemplars": [
            "How do I start a return",
            "I want to return an item from my order",
            "I'd like to send something back",
        ],
        "required_slots": ["order_id"],
    },
    "check_refund_status": {
        "label": "Refund Status",
        "description": "Check the processing status of a refund for a specific order.",
        "exemplars": [
            "How long does a refund take",
            "My refund hasn't shown up yet",
            "Where is my refund",
        ],
        "required_slots": ["order_id"],
    },
    "exchange_item": {
        "label": "Exchange Item",
        "description": "Begin an exchange (different size/product) for an item on a specific order.",
        "exemplars": [
            "Can I exchange this for a different size",
            "I want to swap this item for something else",
            "Can you exchange it instead of refunding me",
        ],
        "required_slots": ["order_id"],
    },
    "check_current_promotions": {
        "label": "Current Promotions",
        "description": "List discount codes/promotions currently active.",
        "exemplars": [
            "Do you have any discount codes right now",
            "Are there any promotions running at the moment",
            "What deals do you currently have",
        ],
        "required_slots": [],
    },
    "promo_code_issue": {
        "label": "Promo Code Issue",
        "description": (
            "Check why a discount code isn't applying, or whether it can still be applied/stacked."
        ),
        "exemplars": [
            "My promo code isn't working",
            "Can I apply a discount code after I've already ordered",
            "Can I use more than one discount code",
            "The code won't apply at checkout",
        ],
        "required_slots": ["discount_code"],
    },
    "loyalty_program_info": {
        "label": "Loyalty Program",
        "description": "Look up loyalty/rewards program details or a customer's points balance.",
        "exemplars": [
            "Do you have a loyalty program",
            "How do I redeem my rewards points",
            "How many points do I have",
            "Do you have a points system for repeat customers",
            "Is there a rewards program I can join",
        ],
        "required_slots": [],
    },
    "price_match_request": {
        "label": "Price Match Request",
        "description": "Submit a price-match request against a competitor's listing.",
        "exemplars": [
            "Do you price match other retailers",
            "I found this cheaper somewhere else, can you match it",
        ],
        "required_slots": [],
    },
    "account_management": {
        "label": "Account Management",
        "description": (
            "Handle an account-management request: password reset, email change, deletion, order "
            "history, verification email, saved addresses."
        ),
        "exemplars": [
            "How do I reset my password",
            "I need to update the email on my account",
            "How do I delete my account",
            "Can I see my past orders",
            "I never got my verification email",
            "Can I save more than one shipping address",
        ],
        "required_slots": [],
    },
    "website_technical_issue": {
        "label": "Website Issue",
        "description": "File a website/technical bug report (checkout, cart, form fields, missing emails).",
        "exemplars": [
            "The checkout page keeps freezing",
            "I can't add anything to my cart",
            "The discount code field won't accept input",
            "I'm not getting order confirmation emails",
        ],
        "required_slots": [],
    },
    "gift_card_inquiry": {
        "label": "Gift Card Inquiry",
        "description": "Handle gift-card purchase, balance check, gift-wrapping, or expiry questions.",
        "exemplars": [
            "Can I buy a gift card",
            "How do I check my gift card balance",
            "Can I send this order as a gift with gift wrapping",
            "Do gift cards expire",
            "Can I put money on a gift card as a present for someone",
        ],
        "required_slots": [],
    },
    "request_human_agent": {
        "label": "Request Human Agent",
        "description": "Transfer the conversation to a human agent.",
        "exemplars": [
            "Can I talk to a real person",
            "I want to speak with a human agent",
            "Can you transfer me to an agent",
        ],
        "required_slots": [],
    },
    "file_complaint": {
        "label": "File a Complaint",
        "description": "Log a customer complaint/escalation, optionally tied to an order.",
        "exemplars": [
            "I want to file a complaint about my order",
            "I'm really unhappy with how this was handled",
            "I'd like to escalate this issue",
            "This experience has been terrible and I want it looked into",
            "I want to raise a formal complaint",
        ],
        "required_slots": [],
    },
    "lookup_crm_record": {
        "label": "CRM Record Lookup",
        "description": (
            "Search Zoho CRM for the customer by email: leads first, then contacts (existing "
            "customers). A record's Owner is the account rep assigned to it."
        ),
        "remote": zoho.LEAD_SEARCH,
        "exemplars": [
            "Can you pull up my account using my email",
            "Look up my info using my email",
            "Can you check if I'm already in your system",
        ],
        "required_slots": ["email"],
    },
    "lookup_by_phone": {
        "label": "Phone Lookup",
        "description": (
            "Search Zoho CRM for the caller by phone number: contacts (existing customers) first, "
            "then leads. A record's Owner is the account rep assigned to it."
        ),
        "remote": zoho.PHONE_SEARCH,
        "exemplars": [
            "Can you look me up by my phone number",
            "Can you find my account with my phone number",
            "My phone number is 555-555-5555, can you pull me up",
            "Search for me using the number I'm calling from",
        ],
        "required_slots": ["phone"],
    },
    "lookup_company_account": {
        "label": "Company Account Lookup",
        "description": (
            "Search Zoho CRM accounts (companies) by name. An account's Owner is the account rep "
            "assigned to it."
        ),
        "remote": zoho.ACCOUNT_SEARCH,
        "exemplars": [
            "I'm calling from Acme Corp, can you pull up our company account",
            "Can you look up the account for my company",
            "I'm calling on behalf of my company",
            "Do you have our business on file",
            "I'm calling from our company, can you pull up our account",
            "Can you pull up our business account",
        ],
        "required_slots": ["company_name"],
    },
    "check_deal_status": {
        "label": "Deal Status",
        "description": (
            "Look up a sales deal in Zoho CRM by name: its stage, amount, expected closing date and "
            "next step. A deal's Owner is the sales rep handling it."
        ),
        "remote": zoho.DEAL_SEARCH,
        "exemplars": [
            "What's the status of our deal",
            "Where are we on the deal",
            "Has the deal closed yet",
            "What stage is our deal in",
            "How much is the deal worth",
        ],
        "required_slots": ["deal_name"],
    },
}
