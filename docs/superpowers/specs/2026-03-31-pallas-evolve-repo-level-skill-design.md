# Convert pallas-evolve Plugin to Repo-Level Local Plugin

**Date:** 2026-03-31
**Status:** Draft

## Problem

The pallas-evolve skills are currently distributed via a user-level marketplace (`primatrix-skills`), requiring each user to:
1. Have the marketplace directory at a specific local path
2. Manually configure `~/.claude/settings.json` with marketplace and plugin references

This creates friction for new contributors who clone the repo and want to use the kernel optimization workflow.

## Solution

Use Claude Code's local plugin mechanism to reference the existing plugin directory from a project-level `.claude/settings.json`. The plugin files stay in place; only the discovery mechanism changes.

## Changes

### 1. Create `.claude/settings.json`

New file in the existing `.claude/` directory (which already contains `worktrees/`):

```json
{
  "plugins": [
    { "type": "local", "path": "./kernel-evolve/plugins/pallas-evolve" }
  ]
}
```

This tells Claude Code to load the plugin from the repo-local path. On first session in the repo, users are prompted to trust the plugin.

### 2. Add Plugin Section to `kernel-evolve/README.md`

The README currently documents the CLI tool but has no mention of the Claude Code plugin. Add a section describing the skill-based workflow:

> ## Claude Code Skills
>
> The pallas-evolve skills are bundled as a local Claude Code plugin. When you open this repo in Claude Code for the first time, you'll be prompted to trust the plugin. After that, invoke skills with `/pallas-evolve:start <config.yaml>`.

### 3. Manual Cleanup (Not Automated)

Users who previously installed via the marketplace should remove `pallas-evolve@primatrix-skills` from their `~/.claude/settings.json` `enabledPlugins` to avoid double-loading.

## What Does NOT Change

- **Plugin directory structure**: `kernel-evolve/plugins/pallas-evolve/` stays as-is
- **SKILL.md files**: All 4 files (start, submit, analyze, reflect) are untouched
- **`.claude-plugin/plugin.json`**: No modifications needed
- **Invocation**: `/pallas-evolve:start`, `/pallas-evolve:submit`, `/pallas-evolve:analyze`, `/pallas-evolve:reflect`
- **Cross-skill references**: Skills reference each other by `pallas-evolve:<name>`, which continues to work

## User Experience

| Step | Before | After |
|------|--------|-------|
| Clone repo | `git clone ...` | `git clone ...` |
| Configure plugin | Edit `~/.claude/settings.json`, add marketplace + enabledPlugins | Nothing - auto-prompted on first use |
| Use skills | `/pallas-evolve:start config.yaml` | `/pallas-evolve:start config.yaml` |

## Verification

Open a new Claude Code session in the repo and confirm `/pallas-evolve:start` appears in available skills.

## Risk Assessment

- **Low risk**: No plugin files are modified. The only new file is `.claude/settings.json`.
- **Backward compatible**: Existing marketplace installs continue to work (users can remove them at their convenience).
- **Trust prompt**: Claude Code's built-in trust mechanism ensures users explicitly approve the plugin before it loads.
