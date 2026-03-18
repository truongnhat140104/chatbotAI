from app.core.router import HotichRouter

router = HotichRouter()

while True:
    q = input("Nhap cau hoi: ").strip()
    if not q or q.lower() in {"exit", "quit"}:
        break

    decision = router.route(q)
    print("\nIntent:", decision.primary_intent)
    print("Scores:", decision.scores)
    print("Reasons:")
    for r in decision.reasons:
        print("-", r)
    print()