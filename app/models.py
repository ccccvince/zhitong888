from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    PARTIAL_FILLED = "PARTIAL_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class MarketMode(str, Enum):
    REAL = "REAL"          # 真实盘中行情 (adata)
    SIMULATED = "SIMULATED"  # 模拟推演行情 (随时可测试撮合与盯盘)


class OrderBookLevel(BaseModel):
    price: float
    volume: int  # 股数


class QuoteSnapshot(BaseModel):
    stock_code: str
    short_name: str
    price: float
    pre_close: float
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    change: float = 0.0
    change_pct: float = 0.0
    volume: float = 0.0      # 成交量(股)
    amount: float = 0.0      # 成交额(元)
    turnover_ratio: float = 0.0  # 换手率(%)
    pe_ratio: float = 0.0        # 市盈率(TTM)
    amplitude: float = 0.0       # 振幅(%)
    limit_up: float = 0.0
    limit_down: float = 0.0
    asks: list[OrderBookLevel] = Field(default_factory=list)  # 卖一..卖五
    bids: list[OrderBookLevel] = Field(default_factory=list)  # 买一..买五
    source: str = "unknown"
    updated_at: str = ""


class MarketIndexSnapshot(BaseModel):
    symbol: str
    code: str
    name: str
    price: float
    pre_close: float
    change: float = 0.0
    change_pct: float = 0.0
    amount: float = 0.0
    updated_at: str = ""


class MinuteBar(BaseModel):
    time: str
    price: float
    avg_price: float = 0.0
    volume: float = 0.0


class OrderRequest(BaseModel):
    stock_code: str = Field(..., min_length=6, max_length=6)
    side: OrderSide
    price: float = Field(..., gt=0)
    quantity: int = Field(..., gt=0)
    strategy_name: str = "MANUAL"


class OrderRecord(BaseModel):
    order_id: str
    stock_code: str
    short_name: str
    side: OrderSide
    price: float
    quantity: int
    filled_quantity: int
    avg_filled_price: float
    frozen_cash: float
    status: OrderStatus
    reject_reason: str = ""
    strategy_name: str = "MANUAL"
    trade_date: str
    created_at: str
    updated_at: str


class TradeRecord(BaseModel):
    trade_id: str
    order_id: str
    stock_code: str
    short_name: str
    side: OrderSide
    price: float
    quantity: int
    amount: float
    commission: float
    transfer_fee: float
    stamp_tax: float
    total_fee: float
    trade_date: str
    created_at: str


class PositionRecord(BaseModel):
    stock_code: str
    short_name: str
    total_qty: int
    available_qty: int
    today_bought_qty: int
    frozen_qty: int
    avg_cost: float
    last_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


class AccountSummary(BaseModel):
    initial_cash: float
    available_cash: float
    frozen_cash: float
    market_value: float
    total_assets: float
    total_pnl: float
    total_pnl_pct: float
    current_trade_date: str
    market_mode: MarketMode


class AlertCondition(str, Enum):
    PRICE_ABOVE = "PRICE_ABOVE"
    PRICE_BELOW = "PRICE_BELOW"
    PCT_ABOVE = "PCT_ABOVE"
    PCT_BELOW = "PCT_BELOW"


class AlertRuleRequest(BaseModel):
    stock_code: str = Field(..., min_length=6, max_length=6)
    condition: AlertCondition
    threshold: float
    cooldown_sec: int = Field(default=300, ge=5, le=86400)
    note: str = ""
    webhook_url: str = ""
    notify_email: bool = False


class AlertRuleRecord(BaseModel):
    rule_id: str
    stock_code: str
    short_name: str
    condition: AlertCondition
    threshold: float
    cooldown_sec: int
    enabled: bool = True
    note: str = ""
    webhook_url: str = ""
    notify_email: bool = False
    last_triggered_at: Optional[str] = None
    created_at: str


class AlertEventRecord(BaseModel):
    event_id: str
    rule_id: str
    stock_code: str
    short_name: str
    message: str
    trigger_price: float
    trigger_pct: float
    webhook_status: str = "NONE"
    email_status: str = "NONE"
    created_at: str


class WatchlistAddRequest(BaseModel):
    stock_code: str = Field(..., min_length=6, max_length=6)
    short_name: str = ""
    group_name: str = "默认"
    is_pinned: bool = False
    note: str = ""


class WatchlistUpdateRequest(BaseModel):
    group_name: Optional[str] = None
    is_pinned: Optional[bool] = None
    note: Optional[str] = None


class RiskConfig(BaseModel):
    max_single_position_pct: float = Field(default=100.0, ge=5.0, le=100.0)
    max_order_amount: float = Field(default=0.0, ge=0.0, le=1_000_000_000.0)
    max_total_loss_pct: float = Field(default=0.0, ge=0.0, le=99.0)
    block_buy_on_loss: bool = True


class ResetAccountRequest(BaseModel):
    initial_cash: float = Field(default=1_000_000.0, ge=10_000.0, le=1_000_000_000.0)


class MarketModeRequest(BaseModel):
    mode: MarketMode


StrategyTypeLiteral = Literal[
    "GRID_MEAN_REVERT",
    "BREAKOUT_BUY",
    "TAKE_PROFIT_STOP_LOSS",
    "MA_CROSS",
    "MACD_CROSS",
]


class AutoStrategyRuleRequest(BaseModel):
    stock_code: str = Field(..., min_length=6, max_length=6)
    strategy_type: StrategyTypeLiteral
    lower_price: float = 0.0
    upper_price: float = 0.0
    fast_period: int = Field(default=5, ge=2, le=60)
    slow_period: int = Field(default=20, ge=3, le=120)
    order_qty: int = Field(default=100, ge=100)
    enabled: bool = True


class AutoStrategyRuleRecord(BaseModel):
    strategy_id: str
    stock_code: str
    strategy_type: str
    lower_price: float
    upper_price: float
    fast_period: int = 5
    slow_period: int = 20
    order_qty: int
    enabled: bool
    last_signal_tag: str = ""
    last_action_at: Optional[str] = None
    created_at: str


class BacktestRequest(BaseModel):
    stock_code: str = Field(..., min_length=6, max_length=6)
    strategy_type: Literal["MA_CROSS", "MACD_CROSS", "GRID"] = "MA_CROSS"
    days: int = Field(default=90, ge=30, le=250)
    initial_cash: float = Field(default=1_000_000.0, ge=10_000.0, le=100_000_000.0)
    position_pct: float = Field(default=95.0, ge=10.0, le=100.0)
    fast_period: int = Field(default=5, ge=2, le=60)
    slow_period: int = Field(default=20, ge=3, le=120)
    grid_lower: float = Field(default=0.0, ge=0.0)
    grid_upper: float = Field(default=0.0, ge=0.0)
    stop_loss_pct: float = Field(default=0.0, ge=0.0, le=50.0)
    take_profit_pct: float = Field(default=0.0, ge=0.0, le=200.0)


class BacktestTradePoint(BaseModel):
    trade_date: str
    side: str
    price: float
    quantity: int
    fee: float
    reason: str
    equity_after: float
    pnl: Optional[float] = None


class BacktestEquityPoint(BaseModel):
    trade_date: str
    close: float
    strategy_assets: float
    benchmark_assets: float
    strategy_return_pct: float
    benchmark_return_pct: float


class BacktestResult(BaseModel):
    stock_code: str
    short_name: str
    strategy_type: str
    days: int
    bar_count: int
    start_date: str
    end_date: str
    initial_cash: float
    final_assets: float
    total_return_pct: float
    benchmark_return_pct: float
    excess_return_pct: float
    max_drawdown_pct: float
    win_rate_pct: float
    trade_count: int
    closed_trades: int
    params_summary: str
    equity_curve: list[BacktestEquityPoint]
    signals: list[BacktestTradePoint]


class LimitUpStockItem(BaseModel):
    stock_code: str
    short_name: str
    board: str
    last_price: float
    latest_change_pct: float
    turnover_ratio: float
    amount: float
    limit_up_count: int
    max_consecutive_boards: int
    last_limit_up_date: str
    limit_up_dates: list[str]
    industry: str = ""
    in_watchlist: bool = False


class LimitUpScreenResponse(BaseModel):
    days: int
    trade_dates: list[str]
    total_hits: int
    source: str
    scanned_at: str
    items: list[LimitUpStockItem]


class AIConfigUpdateRequest(BaseModel):
    provider_name: str = Field(default="通用 OpenAI 兼容", min_length=1, max_length=60)
    base_url: str = Field(default="http://127.0.0.1:11434/v1", min_length=1, max_length=500)
    model: str = Field(default="qwen2.5:14b", min_length=1, max_length=120)
    api_key: Optional[str] = None
    clear_api_key: bool = False
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1500, ge=128, le=8192)
    timeout_sec: int = Field(default=30, ge=3, le=180)
    system_prompt: str = Field(default="", max_length=2000)
    enabled: bool = True


class AIConfigView(BaseModel):
    provider_name: str
    protocol: str = "OpenAI Chat Completions (/chat/completions)"
    base_url: str
    scheme: str
    host: str
    port: int
    base_path: str
    chat_endpoint: str
    models_endpoint: str
    model: str
    has_api_key: bool
    api_key_masked: str
    temperature: float
    max_tokens: int
    timeout_sec: int
    system_prompt: str
    enabled: bool
    last_check_status: str = "UNCONFIGURED"
    last_check_message: str = ""
    last_checked_at: Optional[str] = None
    updated_at: Optional[str] = None
    allow_private_endpoints: bool = True
    platform_trial_available: bool = False
    trial_daily_limit: int = 0
    trial_remaining_today: int = 0
    platform_trial_model: str = ""


class AIConfigTestRequest(BaseModel):
    provider_name: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    clear_api_key: bool = False
    timeout_sec: Optional[int] = Field(default=None, ge=3, le=180)
    probe_chat: bool = True


class AIHealthCheckResponse(BaseModel):
    endpoint_info: AIConfigView
    tcp_reachable: bool
    tcp_latency_ms: Optional[float] = None
    http_status: Optional[int] = None
    chat_probe_ok: Optional[bool] = None
    discovered_models: list[str] = Field(default_factory=list)
    status: str
    message: str
    checked_at: str


AITaskTypeLiteral = Literal[
    "CHAT",
    "DIAGNOSE_STOCK",
    "SCREENER_SUMMARY",
    "DAILY_REVIEW",
    "PARSE_BACKTEST_OR_RULE",
]


class AIActionDraft(BaseModel):
    kind: Literal["BACKTEST", "STRATEGY", "ALERT", "ORDER_PREFILL"]
    title: str
    summary: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AIAskRequest(BaseModel):
    task_type: AITaskTypeLiteral = "CHAT"
    prompt: str = Field(default="", max_length=2000)
    stock_code: Optional[str] = None
    screener_days: int = Field(default=5, ge=1, le=20)
    include_account: bool = True


class AIAskResponse(BaseModel):
    ok: bool
    source: str
    model_used: str
    endpoint_label: str
    latency_ms: float
    reply: str
    drafts: list[AIActionDraft] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str


class AuthUserView(BaseModel):
    user_id: str
    username: str
    display_name: str = ""
    email: str = ""
    email_verified: bool = False
    alert_email_enabled: bool = True
    role: Literal["ADMIN", "USER"] = "USER"
    is_active: bool = True
    ai_calls_today: int = 0
    ai_daily_limit: int = 0
    created_at: str = ""
    last_login_at: Optional[str] = None


class AuthStatusResponse(BaseModel):
    authenticated: bool
    auth_required: bool
    open_register: bool
    invite_required: bool = False
    email_verification_required: bool = False
    email_service_configured: bool = False
    user: Optional[AuthUserView] = None


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=120)
    password: str = Field(..., min_length=1, max_length=128)


class AuthRegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=32)
    password: str = Field(..., min_length=6, max_length=128)
    display_name: str = Field(default="", max_length=32)
    email: str = Field(default="", max_length=120)
    email_code: str = Field(default="", max_length=16)
    invite_code: str = Field(default="", max_length=64)


class SendEmailCodeRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=120)
    purpose: Literal["REGISTER", "RESET_PASSWORD", "BIND_EMAIL"] = "REGISTER"


class ResetPasswordEmailRequest(BaseModel):
    email: str = Field(..., min_length=5, max_length=120)
    code: str = Field(..., min_length=4, max_length=16)
    new_password: str = Field(..., min_length=6, max_length=128)


class BindEmailRequest(BaseModel):
    email: str = Field(default="", max_length=120)
    code: str = Field(default="", max_length=16)
    alert_email_enabled: Optional[bool] = None


class AdminEmailSettingsRequest(BaseModel):
    from_email: str = Field(..., min_length=3, max_length=120)
    email_verification_required: bool = True
    invite_required: bool = False


class AdminEmailTestRequest(BaseModel):
    to_email: str = Field(..., min_length=5, max_length=120)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1, max_length=128)
    new_password: str = Field(..., min_length=6, max_length=128)


class InviteCreateRequest(BaseModel):
    max_uses: int = Field(default=5, ge=0, le=10000)
    count: int = Field(default=1, ge=1, le=20)
    note: str = Field(default="", max_length=80)
    custom_code: str = Field(default="", max_length=32)


class InviteCodeRecord(BaseModel):
    code: str
    created_by: str
    max_uses: int
    used_count: int
    note: str = ""
    enabled: bool = True
    created_at: str


class TogglePayload(BaseModel):
    enabled: bool

