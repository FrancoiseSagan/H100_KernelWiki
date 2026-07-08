# H100 KernelWiki Skill

Codex skill for H100/SM90A kernel programming, focused on TMA, WGMMA, mbarrier, proxy fences, and cluster DSM.

Entry point: `h100-sm90a-kernel-wiki/SKILL.md`.

## Import into Codex

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R h100-sm90a-kernel-wiki "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Then start Codex and invoke it as `$h100-sm90a-kernel-wiki`.
