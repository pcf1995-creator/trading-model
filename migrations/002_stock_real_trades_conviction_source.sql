-- Migration 002: separate conviction-tab from per-ticker tab on stock_real_trades.
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
-- Safe to re-run; idempotent. Run via the Supabase SQL editor.

-- Step 1: drop any existing CHECK constraint on stock_real_trades.source.
-- We look it up by table + column rather than hard-coding a name (the auto-
-- generated constraint name may not match what we expect).
DO $$
DECLARE
    cons_name text;
BEGIN
    FOR cons_name IN
        SELECT con.conname
        FROM pg_constraint con
        JOIN pg_class    cls ON cls.oid = con.conrelid
        JOIN pg_namespace ns ON ns.oid = cls.relnamespace
        WHERE ns.nspname = 'public'
          AND cls.relname = 'stock_real_trades'
          AND con.contype = 'c'
          AND pg_get_constraintdef(con.oid) ILIKE '%source%'
    LOOP
        EXECUTE format('ALTER TABLE public.stock_real_trades DROP CONSTRAINT %I', cons_name);
    END LOOP;
END $$;

-- Step 2: add the new CHECK that includes 'conviction'.
ALTER TABLE public.stock_real_trades
    ADD CONSTRAINT stock_real_trades_source_check
    CHECK (source IN ('scan', 'paper_promotion', 'manual', 'conviction'));

-- Step 3: relabel existing 'manual' rows (all from the conviction Real-All
-- button) to 'conviction' so they appear on the Benchmark tracker.
UPDATE public.stock_real_trades
   SET source = 'conviction'
 WHERE source = 'manual';

-- Sanity check — should return rows grouped by source with counts.
SELECT source, count(*) FROM public.stock_real_trades GROUP BY source ORDER BY source;
