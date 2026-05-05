-- Add 'conviction' to the allowed source values for stock_real_trades.
--
-- Background: the conviction-model tab on the Stocks app (S&P Benchmark L/S)
-- and the per-ticker probability-model tab (Stocks Dashboard) write to the
-- same stock_real_trades table. We want each tab's tracker to show only its
-- own model's trades, which means we need a source value that distinguishes
-- conviction-originated rows from per-ticker-scan rows.
--
-- Schema before:
--   CHECK (source IN ('scan', 'paper_promotion', 'manual'))
-- Schema after:
--   CHECK (source IN ('scan', 'paper_promotion', 'manual', 'conviction'))
--
-- Also relabels any existing rows with source='manual' to source='conviction',
-- because every 'manual' row in the wild today came from the conviction Real-All
-- button (the per-ticker tab only ever writes 'scan').
--
-- Run via the Supabase SQL editor.

ALTER TABLE public.stock_real_trades
  DROP CONSTRAINT IF EXISTS stock_real_trades_source_check;

ALTER TABLE public.stock_real_trades
  ADD CONSTRAINT stock_real_trades_source_check
  CHECK (source IN ('scan', 'paper_promotion', 'manual', 'conviction'));

UPDATE public.stock_real_trades
  SET source = 'conviction'
  WHERE source = 'manual';
