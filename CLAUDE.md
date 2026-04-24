# Claude Code Guidelines

## Commits

**Always commit code changes before considering work complete.** This is especially important for:
- App modifications (Streamlit dashboard changes must be committed for server to pick them up)
- Any changes to Python files that are actively running
- Bug fixes and feature updates

When making code changes, commit immediately after verifying they work, rather than leaving them uncommitted.

## Code Changes

When editing `app.py` or other active files:
1. Make the change
2. Test it (refresh browser, check output)
3. **Commit the change** with a clear message
4. Only then consider the task complete

If you make changes and don't commit them, the changes may not be picked up by the running server or may be lost.
