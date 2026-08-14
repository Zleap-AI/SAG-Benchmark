# Contributing

Thank you for contributing to SAG Benchmark. Before starting substantial work, open an issue or discussion describing the problem and the proposed scope so that implementation effort is aligned with the project direction.

## Development setup

Follow the [development guide](docs/development.md) to install dependencies, configure local services, and run the validation commands. Use `.env.example` as the template; keep credentials and machine-specific settings out of Git.

## Pull requests

A pull request should:

- explain the motivation and user-visible behavior;
- keep the change focused and avoid unrelated formatting churn;
- include tests for changed behavior where practical;
- update the relevant README or `docs/` page;
- report the exact validation commands and any unavailable external service;
- avoid committing generated outputs, credentials, private paths, or local infrastructure details.

Use a clear imperative title, and call out breaking changes or changes to benchmark reproducibility explicitly. Reviewers may request a smaller scope when a change mixes refactoring, behavior changes, and generated artifacts.

## Issues and discussions

Use issues for reproducible bugs, documentation corrections, and focused feature requests. Include the repository revision, operating system, Python version, command, relevant configuration names (with secrets removed), logs, and a minimal reproduction. Use discussions for design questions and experiment proposals when available.

By participating, you agree to follow the project's [Code of Conduct](CODE_OF_CONDUCT.md).
