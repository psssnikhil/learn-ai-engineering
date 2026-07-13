# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| main (live site) | ✅ |

## Reporting a vulnerability

If you discover a security issue in the handbook site, build pipeline, or repository:

1. **Do not** open a public issue for sensitive vulnerabilities.
2. Email or DM the repository owner via [GitHub profile](https://github.com/psssnikhil).
3. Include steps to reproduce and impact assessment.

We will acknowledge within 7 days and work on a fix when appropriate.

## Lessons and exercises

- **Never commit API keys**, `.env` files, or real credentials.
- Use environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) for exercises.
- Report lesson content that encourages unsafe practices (e.g. disabling auth in production examples).

## Dependencies

Report vulnerable dependencies via GitHub Dependabot alerts or a private report to maintainers.
