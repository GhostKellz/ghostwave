# Resolved Advisories

Security advisories that previously appeared in `cargo audit` output and have
since been cleared by dependency updates or feature changes.

| Advisory | Crate | Issue | Resolved by | Date |
|----------|-------|-------|-------------|------|
| RUSTSEC-2021-0139 | `ansi_term` 0.12.1 | Crate unmaintained | Bumped `nnnoiseless` 0.3 → 0.5 with `default-features = false`, dropping its `clap` 2.x / `atty` / `ansi_term` CLI stack | 2026-06-17 |

## How the `ansi_term` advisory was cleared

`nnnoiseless` previously enabled a binary CLI by default, which pulled in
`clap` 2.34 → `ansi_term` 0.12.1 and `atty`. GhostWave only needs the library,
so the dependency was changed to:

```toml
nnnoiseless = { version = "0.5", default-features = false, features = ["dasp"] }
```

This removes the entire CLI dependency chain. After the change `cargo audit`
no longer reports `ansi_term`; the only remaining warnings come from the
`nih-plug` VST framework (see [accepted.md](accepted.md)).
