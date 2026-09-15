# Intent catalogue: exemplar utterances + required slots per intent.
#
# Embedding retrieval compares an incoming utterance against these exemplars
# (not against the intent names) and returns the intent whose exemplars are
# the closest match. Add more exemplar phrasings to improve recall.
#
# Domain: online sales (general e-commerce retailer). Derived from 74
# customer-support FAQs clustered into 24 intents -- see FAQ_INTENTS.md for
# the full FAQ list and the reasoning behind each cluster/split.


# TODO - stage5: Learning mechansim via LLM at the end of day, Human-in-the-loop
INTENTS = {
    "track_order": {
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
        "exemplars": [
            "Can I change my delivery date",
            "Can you reschedule my delivery to next week",
            "I need my package delivered on a different day",
            "Can we move the delivery date",
        ],
        "required_slots": ["order_id", "new_date"],
    },
    "report_delivery_problem": {
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
        "exemplars": [
            "My card was declined at checkout",
            "Why didn't my payment go through",
            "I can't get my card to work on your site",
            "Checkout keeps rejecting my payment",
        ],
        "required_slots": [],
    },
    "payment_info": {
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
        "exemplars": [
            "I was charged twice for one order",
            "The charge on my statement doesn't match my order total",
            "I think I've been overcharged",
            "There's an incorrect charge on my account",
        ],
        "required_slots": ["order_id"],
    },
    "request_invoice": {
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
        "exemplars": [
            "How do I start a return",
            "I want to return an item from my order",
            "I'd like to send something back",
        ],
        "required_slots": ["order_id"],
    },
    "check_refund_status": {
        "exemplars": [
            "How long does a refund take",
            "My refund hasn't shown up yet",
            "Where is my refund",
        ],
        "required_slots": ["order_id"],
    },
    "exchange_item": {
        "exemplars": [
            "Can I exchange this for a different size",
            "I want to swap this item for something else",
            "Can you exchange it instead of refunding me",
        ],
        "required_slots": ["order_id"],
    },
    "check_current_promotions": {
        "exemplars": [
            "Do you have any discount codes right now",
            "Are there any promotions running at the moment",
            "What deals do you currently have",
        ],
        "required_slots": [],
    },
    "promo_code_issue": {
        "exemplars": [
            "My promo code isn't working",
            "Can I apply a discount code after I've already ordered",
            "Can I use more than one discount code",
            "The code won't apply at checkout",
        ],
        "required_slots": ["discount_code"],
    },
    "loyalty_program_info": {
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
        "exemplars": [
            "Do you price match other retailers",
            "I found this cheaper somewhere else, can you match it",
        ],
        "required_slots": [],
    },
    "account_management": {
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
        "exemplars": [
            "The checkout page keeps freezing",
            "I can't add anything to my cart",
            "The discount code field won't accept input",
            "I'm not getting order confirmation emails",
        ],
        "required_slots": [],
    },
    "gift_card_inquiry": {
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
        "exemplars": [
            "Can I talk to a real person",
            "I want to speak with a human agent",
            "Can you transfer me to an agent",
        ],
        "required_slots": [],
    },
    "file_complaint": {
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
        "exemplars": [
            "Can you pull up my account using my email",
            "Look up my info using my email",
            "Can you check if I'm already in your system",
        ],
        "required_slots": ["email"],
    },
}
