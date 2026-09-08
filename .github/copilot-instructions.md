# Copilot instructions — mojagg

Read `AGENTS.md` at the repo root first — it is the authoritative project
guide (build/test/lint commands via pixi, repo layout, performance rules,
testing contract).

Before writing or modifying any Mojo kernel, driver, or binding, read
`.claude/skills/mojagg/SKILL.md` — it is the implementation spec (function
catalog, parity semantics vs numbagg, mandatory performance patterns, Mojo
1.0 idioms and known toolchain pitfalls). Copilot does NOT auto-load
`.claude/skills/` — always read SKILL.md explicitly.

Session handoff with full design history: `HANDOFF.md`.
