# Accepted Advisories

Security advisories that are knowingly accepted because they cannot currently be
removed (a vulnerable crate pulled only transitively with no upstream fix
available). Each entry here should have a matching ID in the `[advisories]
ignore` list in `deny.toml` so the CI gate and this document stay in sync.

All accepted advisories below are confined to the **`ghostwave-vst`** plugin,
which depends on the `nih-plug` framework (a git dependency). They do **not**
affect the `ghostwave-core` library or the `ghostwave` standalone binary.

| Advisory | Crate | Severity | Source chain | Rationale | Review date |
|----------|-------|----------|--------------|-----------|-------------|
| RUSTSEC-2024-0375 | `atty` 0.2.14 | Warning (unmaintained) | `atty` → `nih_log` 0.3.1 → `nih_plug` → `ghostwave-vst` | `atty` is pulled in transitively by `nih_log`, the logger used by the `nih-plug` VST/CLAP framework. There is no released `nih-plug` version that drops `atty`; the crate is a build-time dependency of the plugin only. | 2026-06-17 |
| RUSTSEC-2021-0145 | `atty` 0.2.14 | Warning (unsound) | `atty` → `nih_log` 0.3.1 → `nih_plug` → `ghostwave-vst` | Potential unaligned read on Windows. Same transitive source as above; not reachable in the Linux audio path GhostWave targets. | 2026-06-17 |

## Process for accepting a new advisory

1. Confirm the advisory cannot be cleared by `cargo update` or a dependency
   feature change (e.g. `default-features = false`).
2. Add the `RUSTSEC-XXXX-XXXX` ID to `[advisories] ignore` in `deny.toml`.
3. Add a row to the table above with the rationale and a review date.

## Clearing these advisories

Both `atty` advisories disappear once `nih-plug` (or its `nih_log` dependency)
migrates off `atty` to `std::io::IsTerminal`. Track upstream
[nih-plug](https://github.com/robbert-vdh/nih-plug) and re-test with
`cargo audit` after each framework bump; move cleared entries to
[resolved.md](resolved.md).
