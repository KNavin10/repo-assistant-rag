# Project B Security Review

Review date: 2026-09-09

## Scope and safety boundary

This review used only the synthetic repositories under `tests/fixtures` and
temporary directories created by the offline evaluator. It did not open, load,
send, or trace a real/private repository. The credential-shaped fixture
`tests/fixtures/security_repo/credentials.json` contains only the text
`credential-shaped filename with no real secret`; no real credential was read.

The evaluator uses injected local model functions and makes no provider or
network call. For the tracing check it temporarily set
`LANGSMITH_TRACING=true`, called the local consent guard, received the rejection
shown below, and restored the prior environment value. It did not create a
LangSmith trace.

## Recorded attack results

| Requested attack | Actual input | Actual output |
| --- | --- | --- |
| Traverse outside the repository | `../../outside.py` | `ValueError: File path must not contain '..' traversal.` |
| Select a sensitive credential file | `credentials.json` in the synthetic fixture | Loader indexed only `['src/safe.py', 'src/untrusted.md']`; direct selection returned `PermissionError: Sensitive file selection is not allowed: credentials.json` |
| Request an excessive `k` | `1000000` | `ValueError: k must be between 1 and 8.` |
| Put malicious instructions inside fixture source | `tests/fixtures/security_repo/src/untrusted.md` asks for a shell tool and deletion outside the repository | `status=completed; model_calls=1; action_calls=0; citations=['src/untrusted.md:1-4']` |
| Ask an unsupported question | `What is the weather today?` | `route=unsupported; status=error; model_calls=0` |
| Force a fabricated citation | `src/missing.py:1-2` | `status=error; errors=['Citation does not correspond to any retrieved chunk: src/missing.py:1-2']` |
| Confirm real/private repositories are not traced by default | Project B root with `LANGSMITH_TRACING=true` and no non-fixture consent | `PermissionError: LangSmith tracing is enabled for a non-fixture repository. Disable tracing or pass --allow-non-fixture-tracing explicitly.` |

The repository-root boundary was also tested with a temporary external
repository. It returned `PermissionError` because the path was outside the
configured allowed root.

The tracing result is a default-deny consent boundary, not a claim that tracing
is impossible: a user can explicitly opt in with
`--allow-non-fixture-tracing`. No such opt-in was used in this review.

## Recorded offline evaluator output

Command:

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m repo_assistant.security_evaluation
```

Output:

```text
PASS traversal_rejected - input='../../outside.py'; exception=ValueError: File path must not contain '..' traversal.
PASS external_repository_rejected - exception=PermissionError: Repository path is outside the allowed root C:\Users\<USER>\AppData\Local\Temp\security-eval-1z7vihsi\allowed: C:\Users\<USER>\AppData\Local\Temp\security-eval-1z7vihsi\external
PASS top_k_bounded - input=1000000; exception=ValueError: k must be between 1 and 8.
PASS sensitive_files_skipped - indexed_files=['src/safe.py', 'src/untrusted.md']; selection_exception=PermissionError: Sensitive file selection is not allowed: credentials.json
PASS unsupported_question_no_model - question='What is the weather today?'; route=unsupported; status=error; model_calls=0
PASS fabricated_citation_rejected - citation='src/missing.py:1-2'; status=error; errors=['Citation does not correspond to any retrieved chunk: src/missing.py:1-2']
PASS prompt_injection_is_data - fixture='tests/fixtures/security_repo/src/untrusted.md'; status=completed; model_calls=1; action_calls=0; citations=['src/untrusted.md:1-4']
PASS non_fixture_tracing_rejected - repo='D:\repo-assistant-rag'; LANGSMITH_TRACING=true; exception=PermissionError: LangSmith tracing is enabled for a non-fixture repository. Disable tracing or pass --allow-non-fixture-tracing explicitly.
Security cases: 8/8 passed
```

The machine username in the external-repository output is redacted as `<USER>`.
The temporary directory suffix is generated per run and is not a stable
identifier. All other displayed output is unchanged.

## Secret handling evidence

`git check-ignore -v .env` returned:

```text
.gitignore:2:.env .env
```

`git ls-files --error-unmatch .env` returned:

```text
error: pathspec '.env' did not match any file(s) known to git
```

This verifies that `.env` is ignored and is not tracked, without reading its
contents.

## Test evidence

Before this report was created, the focused run produced `10 passed, 1 skipped,
1 failed`, and the full run produced `86 passed, 1 skipped, 1 failed`. In both
runs, the sole failure was
`test_security_review_records_concrete_evidence`, because
`docs/security-review.md` did not yet exist. The final verification results are
recorded below after the report was added.

- Offline security evaluator: `8/8 passed`.
- Focused loader and security tests: `28 passed in 0.38s`.
- Full provider-free test suite: `88 passed in 13.47s`.
- Python compilation and dependency consistency: completed successfully;
  `No broken requirements found.`

One additional focused rerun without `--basetemp` produced `4 passed, 8 errors`
because Pytest could not access the host's default temporary folder
(`PermissionError: [WinError 5]`). Repeating the same tests with the writable
project-local `--basetemp` produced the successful result above; this was an
environment error, not a failed security assertion.
