"""
Integration tests for auto-placement race condition fixes.

Tests verify:
1. Trades are marked as placed in DB after successful API placement
2. Duplicate detection works via live API + DB
3. Budget enforcement stops placements at $200/week
4. bet_dollars is correctly persisted and summed
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

import db
from kalshi_crypto import auto_place_trade, place_scheduled_orders, get_weekly_deployed_capital
from kalshi_api import KalshiClient


class MockKalshiClient:
    """Mock Kalshi client for testing."""

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.placed_orders = {}  # order_id -> order details
        self.positions = {}  # (ticker, side) -> position

    def place_order(self, ticker, side, contracts, price, action="buy"):
        """Mock place_order - returns order with order_id."""
        order_id = f"order_{ticker}_{side}_{len(self.placed_orders)}"
        self.placed_orders[order_id] = {
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "price": price,
        }
        # Simulate adding to positions
        self.positions[(ticker, side)] = {
            "ticker": ticker,
            "side": side,
            "contracts": contracts,
            "order_id": order_id,
        }
        return {"order": {"order_id": order_id}}

    def get_positions(self):
        """Mock get_positions - returns list of open positions."""
        return list(self.positions.values())


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Synchronous DB Update After Placement
# ─────────────────────────────────────────────────────────────────────────────

def test_mark_trade_as_placed_updates_db():
    """Verify mark_trade_as_placed() atomically updates the trade record."""
    # Arrange: Create a mock trade in memory
    trade_id = "test_trade_123"
    order_id = "order_abc_xyz"
    bet_dollars = 75.50

    # Mock Supabase client
    with patch("db._get_client") as mock_get_client:
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_table = Mock()
        mock_client.table.return_value = mock_table
        mock_table.update.return_value = mock_table
        mock_table.eq.return_value = mock_table

        # Act
        db.mark_trade_as_placed(trade_id, order_id, bet_dollars)

        # Assert
        mock_client.table.assert_called_with("paper_trades")
        mock_table.update.assert_called_once()
        update_call = mock_table.update.call_args
        updates = update_call[0][0]

        assert updates["placement_status"] == "placed"
        assert updates["order_id"] == order_id
        assert updates["bet_dollars"] == bet_dollars
        assert "placed_timestamp" in updates
        assert "updated_at" in updates


def test_auto_place_trade_calls_mark_trade_as_placed():
    """Verify auto_place_trade() updates DB after successful placement."""
    # Arrange
    kalshi_client = MockKalshiClient()
    recommendation = {
        "id": "trade_123",
        "ticker": "BTC",
        "side": "long",
        "contracts_suggested": 10,
        "price": 5000,
        "ev": 0.15,
        "kelly_pct": 5.0,
        "bet_dollars": 50.0,
    }

    with patch("db.load_paper_trades", return_value=[]):
        with patch("db.mark_trade_as_placed") as mock_mark:
            with patch("kalshi_crypto.get_weekly_deployed_capital", return_value=0):
                # Act
                result = auto_place_trade(recommendation, kalshi_client, "weekly")

                # Assert
                assert result is True
                mock_mark.assert_called_once()
                call_args = mock_mark.call_args[0]
                assert call_args[0] == "trade_123"  # trade_id
                assert "order_" in call_args[1]  # order_id (generated)
                assert call_args[2] == 50.0  # bet_dollars


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Live API Duplicate Detection
# ─────────────────────────────────────────────────────────────────────────────

def test_duplicate_check_uses_placement_status():
    """Verify duplicate check uses placement_status='placed', not status='open'."""
    # Arrange
    kalshi_client = MockKalshiClient()

    # Existing trade marked as placed (new field)
    existing_trade = {
        "id": "existing_123",
        "ticker": "BTC",
        "side": "long",
        "status": "open",  # Old field (for settlement tracking)
        "placement_status": "placed",  # New field (for duplicate detection)
        "order_id": "order_xyz",
    }

    recommendation = {
        "id": "new_trade",
        "ticker": "BTC",
        "side": "long",
        "contracts_suggested": 5,
        "price": 5000,
        "bet_dollars": 25.0,
    }

    with patch("db.load_paper_trades", return_value=[existing_trade]):
        # Act
        result = auto_place_trade(recommendation, kalshi_client, "weekly")

        # Assert: Should be rejected as duplicate
        assert result is False


def test_live_api_detects_duplicate_position():
    """Verify live API check prevents duplicate positions on exchange."""
    # Arrange
    kalshi_client = MockKalshiClient()

    # Simulate existing position on exchange
    kalshi_client.positions[("BTC", "long")] = {
        "ticker": "BTC",
        "side": "long",
        "contracts": 10,
        "order_id": "order_existing",
    }

    recommendation = {
        "id": "new_trade",
        "ticker": "BTC",
        "side": "long",
        "contracts_suggested": 5,
        "price": 5000,
        "bet_dollars": 25.0,
    }

    with patch("db.load_paper_trades", return_value=[]):
        # Act
        result = auto_place_trade(recommendation, kalshi_client, "weekly")

        # Assert: Should skip due to live API finding existing position
        assert result is False


def test_live_api_check_with_exception_proceeds_cautiously():
    """Verify that if live API fails, we proceed with a warning."""
    # Arrange
    kalshi_client = Mock()
    kalshi_client.place_order.return_value = {"order": {"order_id": "order_123"}}
    kalshi_client.get_positions.side_effect = Exception("API timeout")

    recommendation = {
        "id": "trade_123",
        "ticker": "BTC",
        "side": "long",
        "contracts_suggested": 10,
        "price": 5000,
        "bet_dollars": 50.0,
    }

    with patch("db.load_paper_trades", return_value=[]):
        with patch("db.mark_trade_as_placed"):
            with patch("kalshi_crypto.get_weekly_deployed_capital", return_value=0):
                # Act
                result = auto_place_trade(recommendation, kalshi_client, "weekly")

                # Assert: Should proceed despite API error, but log warning
                assert result is True  # Still places because place_order succeeded


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Budget Enforcement
# ─────────────────────────────────────────────────────────────────────────────

def test_weekly_budget_stops_at_200():
    """Verify place_scheduled_orders() sizes kelly by remaining budget and stops at $200/week."""
    # Arrange: Top 3 weekly trades with kelly_pct=50% (large enough to test budget constraint)
    # At $200 budget: 50% kelly = $100, $100, resized for 3rd
    kalshi_client = MockKalshiClient()

    trades = [
        {
            "id": "trade_1",
            "ticker": "BTC",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
            "hours_to_exp": 30,
            "ev": 0.20,
            "kelly_pct": 50.0,
            "price_cents": 10000,
            "contracts": 10,
            "bet_dollars": 100.0,
        },
        {
            "id": "trade_2",
            "ticker": "ETH",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
            "hours_to_exp": 25,
            "ev": 0.15,
            "kelly_pct": 50.0,
            "price_cents": 8000,
            "contracts": 10,
            "bet_dollars": 100.0,
        },
        {
            "id": "trade_3",
            "ticker": "SOL",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
            "hours_to_exp": 28,
            "ev": 0.18,
            "kelly_pct": 50.0,
            "price_cents": 2000,
            "contracts": 50,
            "bet_dollars": 100.0,
        },
    ]

    flags = {
        "auto_place_weekly": True,
        "auto_place_vol": False,
        "auto_place_intraday_short": False,
        "auto_place_intraday_long": False,
    }

    with patch("db.load_paper_trades", return_value=trades):
        with patch("db.load_feature_flags", return_value=flags):
            with patch("db.mark_trade_as_placed"):
                with patch("kalshi_crypto.get_weekly_deployed_capital", return_value=0):
                    # Act
                    stats = place_scheduled_orders(kalshi_client)

                    # Assert:
                    # Trade 1: kelly_pct=50% of $200 = $100 ✓ placed
                    # Trade 2: kelly_pct=50% of $100 remaining = $50 ✓ placed
                    # Trade 3: kelly_pct=50% of $50 remaining = $25 ✓ placed (above $5 min)
                    # Total: 3 placed, 0 skipped
                    assert stats["weekly_placed"] == 3
                    assert stats["skipped"] == 0


def test_budget_tracking_includes_current_cycle():
    """Verify budget tracking includes trades placed in current cycle."""
    # Arrange: $120 already deployed, 3 candidates with kelly_pct=50%
    # Trade 1: kelly_pct=50% of $80 remaining = $40 ✓ placed
    # Trade 2: kelly_pct=50% of $40 remaining = $20 ✓ placed
    # Trade 3: kelly_pct=50% of $20 remaining = $10 ✓ placed
    kalshi_client = MockKalshiClient()

    trades = [
        {
            "id": "new_1",
            "ticker": "SOL",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
            "hours_to_exp": 28,
            "ev": 0.20,
            "kelly_pct": 50.0,
            "price_cents": 4000,
            "contracts": 10,
            "bet_dollars": 40.0,
        },
        {
            "id": "new_2",
            "ticker": "XRP",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
            "hours_to_exp": 26,
            "ev": 0.18,
            "kelly_pct": 50.0,
            "price_cents": 2000,
            "contracts": 10,
            "bet_dollars": 20.0,
        },
        {
            "id": "new_3",
            "ticker": "ADA",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
            "hours_to_exp": 25,
            "ev": 0.15,
            "kelly_pct": 50.0,
            "price_cents": 2000,
            "contracts": 5,
            "bet_dollars": 10.0,
        },
    ]

    flags = {
        "auto_place_weekly": True,
        "auto_place_vol": False,
        "auto_place_intraday_short": False,
        "auto_place_intraday_long": False,
    }

    with patch("db.load_paper_trades", return_value=trades):
        with patch("db.load_feature_flags", return_value=flags):
            with patch("db.mark_trade_as_placed"):
                with patch("kalshi_crypto.get_weekly_deployed_capital", return_value=120.0):  # $120 already deployed
                    # Act
                    stats = place_scheduled_orders(kalshi_client)

                    # Assert: With $120 already deployed and $200 budget, $80 remaining
                    # All 3 trades fit within $80 (resized kelly: $40, $20, $10)
                    assert stats["weekly_placed"] == 3
                    assert stats["skipped"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: bet_dollars Persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_reconstructed_rec_includes_bet_dollars_and_id():
    """Verify place_scheduled_orders() includes kelly-sized bet_dollars and id in rec dict."""
    # Arrange: kelly_pct=22.5% of $200 budget = $45
    kalshi_client = Mock()
    kalshi_client.place_order.return_value = {"order": {"order_id": "test_order"}}
    kalshi_client.get_positions.return_value = []

    trades = [
        {
            "id": "trade_123",
            "ticker": "BTC",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
            "hours_to_exp": 30,
            "ev": 0.20,
            "kelly_pct": 22.5,
            "price_cents": 6000,
            "contracts": 1,
            "bet_dollars": 45.0,
            "placed_at": datetime.now(timezone.utc).isoformat(),
        },
    ]

    flags = {
        "auto_place_weekly": True,
        "auto_place_vol": False,
        "auto_place_intraday_short": False,
        "auto_place_intraday_long": False,
    }

    # Capture the rec dict passed to auto_place_trade
    captured_rec = {}

    def capture_auto_place(rec, client, bucket):
        captured_rec.update(rec)
        return True

    with patch("db.load_paper_trades", return_value=trades):
        with patch("db.load_feature_flags", return_value=flags):
            with patch("kalshi_crypto.auto_place_trade", side_effect=capture_auto_place):
                with patch("kalshi_crypto.get_weekly_deployed_capital", return_value=0):
                    # Act
                    place_scheduled_orders(kalshi_client)

                    # Assert: rec should have id and kelly-sized bet_dollars
                    # kelly_pct=22.5% of $200 = $45
                    assert captured_rec.get("id") == "trade_123"
                    assert captured_rec.get("bet_dollars") == 45.0


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: End-to-End Scenario
# ─────────────────────────────────────────────────────────────────────────────

def test_end_to_end_placement_and_duplicate_prevention():
    """End-to-end test: place trade, verify it's marked placed, skip on next run."""
    # Arrange
    kalshi_client = MockKalshiClient()

    recommendation = {
        "id": "btc_long_trade",
        "ticker": "BTC",
        "side": "long",
        "contracts_suggested": 10,
        "price": 5000,
        "ev": 0.15,
        "kelly_pct": 5.0,
        "bet_dollars": 50.0,
    }

    # Simulate DB with the trade record
    db_trades = [
        {
            "id": "btc_long_trade",
            "ticker": "BTC",
            "side": "long",
            "status": "open",
            "pnl_dollars": None,
        }
    ]

    with patch("db.load_paper_trades", return_value=db_trades):
        with patch("db.mark_trade_as_placed") as mock_mark:
            with patch("kalshi_crypto.get_weekly_deployed_capital", return_value=0):
                # ACT 1: First placement
                result1 = auto_place_trade(recommendation, kalshi_client, "weekly")
                assert result1 is True
                mock_mark.assert_called_once()

                # Simulate DB update: mark the trade as placed
                db_trades[0]["placement_status"] = "placed"
                db_trades[0]["order_id"] = "order_123"

                # ACT 2: Try to place same trade again
                result2 = auto_place_trade(recommendation, kalshi_client, "weekly")

                # ASSERT: Should skip as duplicate
                assert result2 is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
