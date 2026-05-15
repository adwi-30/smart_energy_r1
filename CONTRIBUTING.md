# Contributing to Smart Energy RL

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable, production-ready code. Protected. |
| `dev` | Integration branch for new features. |
| `feature/<name>` | Individual feature branches (e.g. `feature/agent`, `feature/api`) |

## How to Contribute

1. **Create a feature branch from `dev`:**
   ```bash
   git checkout dev
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and commit with a descriptive message:**
   ```bash
   git add .
   git commit -m "feat: describe your change"
   ```

3. **Push and open a Pull Request into `dev`:**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a PR on GitHub: `feature/your-feature-name` → `dev`

4. **PR checklist before merging:**
   - [ ] All tests pass (`python -m pytest tests/ -v`)
   - [ ] Code is documented (docstrings added)
   - [ ] Config changes go in `configs/` YAML files — not hardcoded
   - [ ] New experiment results saved to `experiments/` and `logs/`
   - [ ] Reviewer assigned

5. **Merge `dev` into `main`** only after full integration testing.

## Commit Message Convention

| Prefix | Use for |
|---|---|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation update |
| `test:` | Test additions/changes |
| `chore:` | Maintenance (deps, CI config) |
| `refactor:` | Code restructure without behaviour change |

## Experiment Versioning

- Tag every completed experiment: `git tag exp-qlearning-3 -m "α=0.15, ε_decay=0.992"`
- All experiment CSVs and JSON logs must be committed with the tag.
- Rollback: `python scripts/rollback.py <tag-name>`
