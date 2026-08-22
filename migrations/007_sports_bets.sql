-- Migration 007: sports_bets table
-- Tracks EV scanner recommendations for NCAAF + NFL paper trades.
-- Parallel to paper_trades (Kalshi) but for sports markets.

create table if not exists sports_bets (
  id           bigint generated always as identity primary key,
  game_id      text        not null,
  sport        text        not null,          -- 'ncaaf' | 'nfl'
  home         text        not null,
  away         text        not null,
  commence     text,                          -- ISO8601 kickoff time
  market       text        not null,          -- 'total' | 'spread' | 'moneyline'
  side         text        not null,          -- 'Over', 'Under', team name, etc.
  odds         integer     not null,          -- American odds (e.g. -110)
  line         numeric,                       -- spread point or total line
  book         text,                          -- best-odds book
  model_prob   numeric     not null,
  market_prob  numeric     not null,
  edge         numeric     not null,          -- model_prob - market_prob
  ev           numeric     not null,
  kelly_pct    numeric     not null,
  bet_dollars  numeric,
  status       text        not null default 'open',   -- 'open' | 'settled'
  result       text,                          -- 'win' | 'loss' | 'push'
  pnl_dollars  numeric,
  scanned_at   timestamptz not null default now(),
  settled_at   timestamptz
);

-- Prevent double-saving the same recommendation from the same scan run
create unique index if not exists sports_bets_dedup
  on sports_bets (game_id, market, side, date(scanned_at));

-- Fast lookups for open bets and settlement
create index if not exists sports_bets_status on sports_bets (status);
create index if not exists sports_bets_sport  on sports_bets (sport);
create index if not exists sports_bets_commence on sports_bets (commence);
