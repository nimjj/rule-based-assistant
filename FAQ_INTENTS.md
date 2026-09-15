# Online Sales: FAQ → Intent Derivation

Phase 3 deliverable: pick one industry vertical and show the FAQ-to-intent
process end to end, per the pipeline in the root `CLAUDE.md` ("sentence
embeddings, compare against exemplar utterances for each intent"). Industry
chosen: **online sales (general e-commerce retailer)** — ordering, shipping,
payments, returns, product info, promotions, account, and support.

## Process

1. Collect the questions a customer actually asks an online retailer's
   support line/chat — 74 of them below, grouped by topic area.
2. Cluster FAQs that resolve to the *same backend action* (or the same class
   of answer) into one intent. Two FAQs become one intent when a human agent
   would handle them identically once they know which one it is — "is this
   in stock" and "when will it be restocked" both resolve to a stock lookup
   on one product, so they're one intent (`check_product_availability`), not
   two.
3. For each intent, write exemplar utterances covering the different ways a
   customer phrases the same ask (not the FAQ text itself — FAQs are
   third-person/formal, customer utterances are first-person/casual). These
   go in `intents.py` and are what the embedding retriever actually compares
   against (`embeddings.py`).
4. Decide the intent's `required_slots` — what a backend action needs before
   it can fire (`order_id`, `product_name`, `discount_code`, `new_date`).
   Purely informational intents (policy questions, "what payment methods do
   you take") need none.
5. Map each intent to a mock backend tool in `mcp_tools.py`, same shape as
   Stage 3's existing tool catalog.

74 FAQs clustered into **24 intents** (~3 FAQs/intent) — see the table below.
This ratio is what you'd expect from a real vertical: enough intents to route
distinct backend actions, not so many that exemplar sets overlap and the
embedding retriever can't tell them apart.

## FAQ → Intent table

| # | FAQ | Intent |
|---|---|---|
| 1 | Where is my order? | `track_order` |
| 2 | Can you tell me the status of my order? | `track_order` |
| 3 | How do I track my package? | `track_order` |
| 4 | Why hasn't my order shipped yet? | `track_order` |
| 5 | What's the estimated delivery date for my order? | `track_order` |
| 6 | My tracking number isn't updating — is something wrong? | `track_order` |
| 7 | Has my order been delivered yet? | `track_order` |
| 8 | Tracking says "out for delivery" but I haven't received it — what do I do? | `track_order` |
| 9 | Can I get a delivery notification? | `track_order` |
| 10 | Can I cancel my order? | `cancel_order` |
| 11 | How do I cancel a recurring/subscription order? | `cancel_order` |
| 12 | Can I change my order after it's been placed? | `modify_order` |
| 13 | Can I update the shipping address on an order I already placed? | `modify_order` |
| 14 | Can I add another item to an order I just placed? | `modify_order` |
| 15 | Can I change the size or color of an item I ordered? | `modify_order` |
| 16 | Can two separate orders be combined into one shipment? | `modify_order` |
| 17 | Can I change my delivery date? | `change_delivery_date` |
| 18 | Can I reschedule delivery to a different day? | `change_delivery_date` |
| 19 | My package arrived damaged — what do I do? | `report_delivery_problem` |
| 20 | My order says delivered but I never received it. | `report_delivery_problem` |
| 21 | I received the wrong item — what should I do? | `report_delivery_problem` |
| 22 | Part of my order is missing. | `report_delivery_problem` |
| 23 | My package was delivered to the wrong address. | `report_delivery_problem` |
| 24 | Why was my card declined at checkout? | `payment_declined` |
| 25 | What payment methods do you accept? | `payment_info` |
| 26 | Do you accept PayPal / Apple Pay / gift cards? | `payment_info` |
| 27 | Can I split payment across two cards? | `payment_info` |
| 28 | When will my card actually be charged? | `payment_info` |
| 29 | Why was I charged twice for one order? | `billing_dispute` |
| 30 | Why is the charge on my statement different from my order total? | `billing_dispute` |
| 31 | Can I get a copy of my receipt or invoice? | `request_invoice` |
| 32 | Can I get an itemized invoice for my order? | `request_invoice` |
| 33 | Is this item in stock? | `check_product_availability` |
| 34 | When will this item be back in stock? | `check_product_availability` |
| 35 | Do you have this item in a different size or color? | `check_product_availability` |
| 36 | What are the dimensions of this product? | `product_information` |
| 37 | What material is this product made of? | `product_information` |
| 38 | Does this product come with a warranty? | `product_information` |
| 39 | Are there customer reviews for this item? | `product_information` |
| 40 | Can I request a sample before I buy? | `product_information` |
| 41 | Is this product compatible with [x]? | `product_information` |
| 42 | What's your return policy? | `return_policy_info` |
| 43 | How long do I have to return an item? | `return_policy_info` |
| 44 | Do I have to pay for return shipping? | `return_policy_info` |
| 45 | Can I return a sale or clearance item? | `return_policy_info` |
| 46 | Can I return something without the original packaging or tags? | `return_policy_info` |
| 47 | How do I start a return? | `initiate_return` |
| 48 | How long does a refund take to process? | `check_refund_status` |
| 49 | My refund hasn't shown up yet — where is it? | `check_refund_status` |
| 50 | Can I exchange an item for a different size instead of returning it? | `exchange_item` |
| 51 | Can I exchange an item for a completely different product? | `exchange_item` |
| 52 | Do you have any discount codes right now? | `check_current_promotions` |
| 53 | Why isn't my promo code working? | `promo_code_issue` |
| 54 | Can I apply a discount code after I've already placed my order? | `promo_code_issue` |
| 55 | Can I use more than one discount code on an order? | `promo_code_issue` |
| 56 | Do you have a loyalty or rewards program? | `loyalty_program_info` |
| 57 | How do I redeem my loyalty points? | `loyalty_program_info` |
| 58 | Do you price match other retailers? | `price_match_request` |
| 59 | How do I reset my password? | `account_management` |
| 60 | How do I update the email address on my account? | `account_management` |
| 61 | How do I delete my account? | `account_management` |
| 62 | Can I view my past order history? | `account_management` |
| 63 | I never received my account verification email. | `account_management` |
| 64 | Can I save more than one shipping address to my account? | `account_management` |
| 65 | The checkout page keeps freezing or throwing an error. | `website_technical_issue` |
| 66 | I can't add an item to my cart. | `website_technical_issue` |
| 67 | My discount code field won't accept my code. | `website_technical_issue` |
| 68 | I'm not receiving order confirmation emails. | `website_technical_issue` |
| 69 | Can I purchase a gift card? | `gift_card_inquiry` |
| 70 | How do I check my gift card balance? | `gift_card_inquiry` |
| 71 | Can I send my order as a gift with gift wrapping? | `gift_card_inquiry` |
| 72 | Do gift cards expire? | `gift_card_inquiry` |
| 73 | How do I talk to a live person / human agent? | `request_human_agent` |
| 74 | I want to file a complaint about my order experience. | `file_complaint` |

## Notes on borderline calls

- **`website_technical_issue` #67 vs. `promo_code_issue`**: "my discount code
  field won't accept my code" *sounds* like a promo-code question, but the
  fault is the website, not the code — it's a technical-issue report, not a
  request to check/apply a code. Kept separate from `promo_code_issue`
  because the backend action differs (bug report vs. code lookup).
- **`check_current_promotions` vs. `promo_code_issue`**: "do you have any
  codes right now" (browsing) and "my code isn't working" (troubleshooting a
  specific code) read similarly but need different `required_slots`
  (`promo_code_issue` needs the code itself) and different backend calls.
- **`payment_declined` stayed separate from `payment_info`/`billing_dispute`**:
  a decline happens *before* an order exists (no `order_id` yet), whereas
  `billing_dispute` is about a charge on a completed order — different slot
  requirements, different urgency.
- **`file_complaint` has no required slots**, unlike most order-related
  intents — a complaint may not reference a specific order (e.g. "the site
  is impossible to use"), so gating it on `order_id` would strand it in
  `AWAITING_INFO` for complaints that have none.

See `intents.py` for the exemplar utterances and `mcp_tools.py` for the tool
each intent resolves to. `README.md` covers how to actually run this against
sample calls and the labeled test set.
