from __future__ import annotations

from pathlib import Path

from app.engine.alert_engine import AlertEngine
from app.models import AlertCondition, AlertRuleRequest, QuoteSnapshot
from app.storage import Database


def test_alert_triggers_and_respects_cooldown(tmp_path: Path) -> None:
    db = Database(tmp_path / "alert.db")
    engine = AlertEngine(db)

    r1 = engine.add_rule(
        AlertRuleRequest(
            stock_code="600519",
            condition=AlertCondition.PRICE_ABOVE,
            threshold=1500.0,
            cooldown_sec=120,
            note="突破目标位",
        ),
        short_name="贵州茅台",
    )
    r2 = engine.add_rule(
        AlertRuleRequest(
            stock_code="300750",
            condition=AlertCondition.PCT_BELOW,
            threshold=-3.0,
            cooldown_sec=120,
        ),
        short_name="宁德时代",
    )

    q_hit = {
        "600519": QuoteSnapshot(
            stock_code="600519",
            short_name="贵州茅台",
            price=1505.0,
            pre_close=1480.0,
            change=25.0,
            change_pct=1.69,
        ),
        "300750": QuoteSnapshot(
            stock_code="300750",
            short_name="宁德时代",
            price=250.0,
            pre_close=260.0,
            change=-10.0,
            change_pct=-3.85,
        ),
    }

    events1 = engine.evaluate(q_hit)
    assert len(events1) == 2

    # Immediate re-eval must be suppressed by cooldown
    events2 = engine.evaluate(q_hit)
    assert len(events2) == 0

    engine.toggle_rule(r1.rule_id, False)
    rules = engine.list_rules()
    assert any(r.rule_id == r1.rule_id and not r.enabled for r in rules)

    engine.delete_rule(r2.rule_id)
    assert len(engine.list_rules()) == 1
