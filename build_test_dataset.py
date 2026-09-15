# One-off script that (re)generates intent_test_dataset.xlsx for the online
# sales domain: paraphrased utterances per intent (deliberately avoiding the
# exact wording used in intents.py exemplars, so the test measures semantic
# generalization, not keyword overlap) plus a handful of out-of-domain
# "unknown" utterances. Not part of the runtime pipeline -- run once to
# regenerate the dataset, same convention as the rest of Stage 3's xlsx files.

import pandas as pd

ROWS = [
    ("I haven't heard anything about my order since I placed it", "track_order", "No 'status'/'track' keyword"),
    ("Any news on when this is supposed to arrive", "track_order", "'news' instead of 'status'"),
    ("I'd like to back out of this purchase", "cancel_order", "No 'cancel' keyword"),
    ("Please stop the recurring charge and shipment", "cancel_order", "Indirect phrasing for subscription cancel"),
    ("I need to send this to a different address than I gave you", "modify_order", "No 'change order' phrase"),
    ("Can you swap the color on what I bought for something else", "modify_order", "Casual phrasing, no 'change/update'"),
    ("Is there any way to get this here a day later than planned", "change_delivery_date", "No 'reschedule/delivery date' phrase"),
    ("The box showed up smashed on one side", "report_delivery_problem", "No 'damaged' keyword"),
    ("It's marked as delivered but there's nothing on my porch", "report_delivery_problem", "Indirect 'never received'"),
    ("This isn't what I ordered at all", "report_delivery_problem", "No 'wrong item' phrase"),
    ("My card got rejected when I tried to buy this", "payment_declined", "'rejected' instead of 'declined'"),
    ("None of my payment attempts are going through at checkout", "payment_declined", "No 'card declined' phrase"),
    ("Which cards or wallets can I pay with on your site", "payment_info", "No 'payment methods' phrase"),
    ("Do you let people pay with more than one card at once", "payment_info", "Indirect 'split payment'"),
    ("The amount taken from my account doesn't look right", "billing_dispute", "No 'refund/dispute' keyword"),
    ("I think I was billed twice for the same thing", "billing_dispute", "Different phrasing than exemplars"),
    ("Can someone email me proof of what I paid", "request_invoice", "No 'receipt/invoice' keyword"),
    ("I need a breakdown of the charges for my order", "request_invoice", "'breakdown' instead of 'itemized'"),
    ("Do you currently have any of the Marlow Sofa left", "check_product_availability", "'left' instead of 'in stock'"),
    ("When's the Nimbus Jacket coming back", "check_product_availability", "No 'stock' keyword"),
    ("How big is the Aurora Desk Lamp exactly", "product_information", "'how big' instead of 'dimensions'"),
    ("What's the Rowan Backpack made from", "product_information", "'made from' instead of 'material'"),
    ("How many days do I get to send something back", "return_policy_info", "No 'return policy' phrase"),
    ("Will I have to cover the cost of shipping it back", "return_policy_info", "No 'return shipping' phrase"),
    ("I'd like to send this item back for a refund", "initiate_return", "'send back' instead of 'return'"),
    ("Where's the money from my returned item", "check_refund_status", "No 'refund' keyword directly named"),
    ("It's been over a week and I still don't see my money back", "check_refund_status", "Indirect refund-delay phrasing"),
    ("Instead of a refund can I just get a different size sent", "exchange_item", "No 'exchange' keyword"),
    ("Are there any deals I should know about before I check out", "check_current_promotions", "No 'discount/code' keyword"),
    ("The code I have just won't go through at checkout", "promo_code_issue", "No 'promo/discount' keyword"),
    ("Can I stack this code with another one I have", "promo_code_issue", "'stack' instead of 'combine'"),
    ("Do you have some kind of points system for repeat customers", "loyalty_program_info", "No 'loyalty/rewards' keyword"),
    ("I saw this listed cheaper on another site, can you do anything", "price_match_request", "No 'price match' phrase"),
    ("I forgot my password and can't get back in", "account_management", "No 'reset password' phrase"),
    ("Can I look back at everything I've bought before", "account_management", "'everything I've bought' instead of 'order history'"),
    ("The page just hangs whenever I try to pay", "website_technical_issue", "'hangs' instead of 'freezing'"),
    ("Nothing happens when I click add to bag", "website_technical_issue", "No 'cart' keyword"),
    ("Can I put money on a card as a present for someone", "gift_card_inquiry", "No 'gift card' phrase"),
    ("How much is left on the card I was given", "gift_card_inquiry", "No 'balance' keyword"),
    ("Is there a way to get an actual person on the line", "request_human_agent", "No 'agent/human' keyword"),
    ("I want this looked into properly, this experience has been awful", "file_complaint", "No 'complaint' keyword"),
    ("What's the weather like today", "unknown", "Out-of-domain, unrelated to online sales"),
    ("Can you recommend a good restaurant nearby", "unknown", "Out-of-domain, unrelated to online sales"),
    ("What time do you close on weekends", "unknown", "Plausible-sounding but no matching intent (online-only retailer)"),
    ("Tell me a joke", "unknown", "Out-of-domain, nonsensical request"),
    ("Do you sell insurance for my car", "unknown", "Different industry vertical entirely"),
    ("Can you help me file my taxes", "unknown", "Different industry vertical entirely"),
]

df = pd.DataFrame(ROWS, columns=["Utterance", "Expected Intent", "Rule-based Challenge"])
df.insert(0, "ID", range(1, len(df) + 1))
df.to_excel("intent_test_dataset.xlsx", index=False)
print(f"Wrote {len(df)} rows to intent_test_dataset.xlsx")
