# Security policy

## Reporting a vulnerability

Please report suspected security vulnerabilities privately to the project maintainers through the security contact listed in the repository profile. Do not disclose an exploitable vulnerability in a public issue before maintainers have had an opportunity to investigate it.

Include, when safe to do so:

- the affected revision or release;
- a concise description of the impact;
- reproduction steps or a minimal proof of concept;
- logs or stack traces with secrets removed;
- any suggested mitigation.

Do not send API keys, passwords, access tokens, private URLs, or personal data in the report. Rotate any credential that may have been exposed before reporting it.

## Scope and support

This project is under active development. Security fixes are evaluated according to impact and affected versions; maintainers will communicate a fix or mitigation through the private report and an appropriate public release note when available.

## Safe reporting practices

The repository includes a credential guard for Git remotes. Run it before publishing changes, and review the complete diff for secrets and machine-specific configuration. Keep `.env` files, generated outputs, local databases, and service credentials untracked.


## Trusted artifact boundary

External methods may load Python pickle files, HippoRAG serialized indexes, or
Hyper-RAG `.hgdb` databases. These formats can execute code or instantiate
unexpected objects during deserialization. Load them only from a trusted local
run produced by the same repository checkout and user account.

Do not load artifacts received from issues, pull requests, public URLs, shared
buckets, or unknown users. Generated paths must remain under the selected
method's `caches/` or `outputs/` root. Treat an artifact with an unexpected
owner, symlink, checksum, or path traversal as untrusted and regenerate it.
