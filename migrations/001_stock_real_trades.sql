-- Real-trades tracker for the S&P L/S model.
-- Mirrors stock_paper_trades, plus a `source` column so we know if a row came
-- straight from a scan or was promoted from a paper trade, and a nullable
-- pointer back to the originating paper row.
--
-- Run via the Supabase SQL editor.

create table if not exists public.stock_real_trades (
  id              text primary key,
  ticker          text not null,
  side            text not null check (side in ('long', 'short')),
  entry_price     numeric not null,
  entry_date      date   not null,
  shares          numeric not null,
  dollars         numeric,
  model_prob      numeric,
  status          text   not null default 'open' check (status in ('open', 'closed')),
  exit_reason     text,
  exit_price      numeric,
  exit_date       date,
  pnl_dollars     numeric,
  pnl_pct         numeric,
  placed_at       timestamptz not null default now(),
  source          text   not null default 'scan' check (source in ('scan', 'paper_promotion', 'manual')),
  paper_trade_id  text,
  notes           text,
  updated_at      timestamptz not null default now()
);

create index if not exists stock_real_trades_status_idx
  on public.stock_real_trades (status);

create index if not exists stock_real_trades_paper_id_idx
  on public.stock_real_trades (paper_trade_id);
