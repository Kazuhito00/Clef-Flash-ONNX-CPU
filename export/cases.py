# -*- coding: utf-8 -*-
"""The requests the oracle (export/make_golden.py) and the gates (verify/) both use.

Chosen to hit every branch of the layout:
  noul / choice / score, several questions in one request, a JSON (non-string) state, choice ids that are
  not already in sorted order, noul with custom criteria, non-ASCII text, a state long enough that the
  sequence pads to a different multiple of 64.
"""

LONG_STATE = (
    "Ticket #48213, opened Monday. The customer writes: our nightly export has been failing since the "
    "update on Friday. The job times out after 30 minutes, and the log shows repeated 502 responses from the "
    "storage gateway. We tried restarting the worker twice and rotating the credentials. Support replied on "
    "Tuesday asking for the job id, which we sent. No further reply. The finance team needs the report by "
    "Thursday morning for the quarterly close, and the CEO has asked for a status twice already. "
) * 3

CASES = {
    "choice-basic": {
        "state": "I was charged twice for my Pro subscription this month, please fix.",
        "questions": {"team": {"type": "choice", "instructions": "Which team should handle this?",
                               "criteria": {"billing": "charges, refunds, invoices",
                                            "technical support": "bugs, outages, errors",
                                            "sales": "pricing, upgrades, new contracts"}}}},
    "noul-basic": {
        "state": "Thanks so much for the quick fix yesterday, everything works great now!",
        "questions": {"angry": {"type": "noul", "instructions": "The customer is angry"}}},
    "score-basic": {
        "state": "URGENT: production database is down, all customers affected.",
        "questions": {"urgency": {"type": "score", "instructions": "How urgent is this?",
                                  "criteria": ["low", "medium", "high", "critical"]}}},
    "multi": {
        "state": "This is the third time this week my payouts failed and nobody answers my emails. Unacceptable.",
        "questions": {
            "angry": {"type": "noul", "instructions": "The customer is angry"},
            "team": {"type": "choice", "instructions": "Which team?",
                     "criteria": {"billing": "payments", "technical": "bugs"}},
            "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]}}},
    "json-state": {
        "state": {"order": {"id": 1042, "status": "shipped", "items": ["lamp", "desk"]}, "customer": "Aiko",
                  "notes": None},
        "questions": {"intent": {"type": "choice", "instructions": "What does the customer want next?",
                                 "criteria": {"track": "see where the parcel is", "cancel": "cancel the order",
                                              "return": "send the items back"}}}},
    "unsorted-keys": {
        "state": "The app crashes every time I open the settings screen on Android 14.",
        "questions": {"kind": {"type": "choice", "instructions": "What kind of report is this?",
                               "criteria": {"zebra": "a joke", "bug": "a defect", "Alpha": "a feature request",
                                            "bug2": "a duplicate defect"}}}},
    "noul-custom": {
        "state": "The deploy finished, all health checks are green and latency is back to normal.",
        "questions": {"incident": {"type": "noul", "instructions": "There is an ongoing incident",
                                   "criteria": {"true": "Something is still broken.",
                                                "false": "Everything is healthy."}}}},
    "non-ascii": {
        "state": "Pro プランの料金が今月 2 回引き落とされています。対応をお願いします。 café ☕",
        "questions": {"team": {"type": "choice", "instructions": "どのチームが対応すべきですか？",
                               "criteria": {"請求": "請求・返金", "技術": "不具合・障害", "営業": "価格・契約"}},
                      "angry": {"type": "noul", "instructions": "顧客は怒っている"}}},
    "long-state": {
        "state": LONG_STATE,
        "questions": {"urgent": {"type": "noul", "instructions": "This needs attention today"},
                      "team": {"type": "choice", "instructions": "Which team?",
                               "criteria": {"infra": "storage and gateways", "finance": "reports", "support": "tickets"}}}},
    "score-eight": {
        "state": "The restaurant was fine, the food arrived cold but the staff apologised and gave us dessert.",
        "questions": {"rating": {"type": "score", "instructions": "How would the guest rate the visit?",
                                 "criteria": ["terrible", "bad", "poor", "so-so", "okay", "good", "great", "excellent"]}}},
}
