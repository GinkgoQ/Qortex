# AGENTS.md

## Core Rules

- Use `rg`, `git grep`, Serena/LSP symbol tools, type information, and existing tests before reading full files.
- Do not read `dist/`, `build/`, `coverage/`, `.venv/`, `node_modules/`, generated SDKs, vendored code, caches, or compiled artifacts unless the task explicitly requires it.
- Before editing, identify the smallest owning package/module and read its local `AGENTS.md` if present.
- Choose the stack per subtask. A valid solution may combine framework code, CLI tools, database queries, static analysis, tests, profiling, tracing, and documentation checks.
- Do not force one tool, framework, pattern, or abstraction across unrelated subtasks.
- Prefer existing architecture, public APIs, extension points, and project conventions over new abstractions.
- Fix root causes, not symptoms. Do not patch only the visible failing case if the underlying failure mode is broader.

## Context and Investigation

- Start from the observable failure, requested behavior, or concrete user story.
- For bugs, reproduce with the narrowest command, test, trace, or runtime scenario before editing.
- For large or ambiguous work, create `.agent/PLAN.md` with scope, assumptions, affected files, risks, and verification steps.
- Keep `.agent/PLAN.md` updated after each meaningful discovery or verification step.
- Inspect call sites, ownership boundaries, data flow, configuration, error handling, and tests before changing behavior.
- Use symbolic navigation first: definitions, references, callers, imports, inheritance, routes, schemas, migrations, and public exports.
- Read only the files needed to prove the change.
- If requirements, runtime behavior, or external API semantics are unclear, verify them before implementing.

## Implementation Standard

- Produce production-grade code for the real codebase, not examples, demos, placeholders, or one-off patches.
- Do not use mockups, fake implementations, hardcoded success paths, fabricated data, or TODO-based behavior.
- Do not bypass validation, permissions, error handling, persistence, logging, retries, transactions, migrations, typing, or tests to make a task appear complete.
- Do not create parallel systems when an existing project mechanism can be extended safely.
- Do not duplicate business logic. Reuse existing domain services, schemas, adapters, validators, and shared utilities.
- Keep changes minimal in surface area but complete in behavior.
- Design for the general failure class, not only the provided input.
- Handle edge cases: empty input, invalid input, missing config, partial failure, concurrency, retries, timeouts, cancellation, large data, and backward compatibility.
- Preserve public APIs unless the task explicitly requires a breaking change.
- When changing public behavior, update tests, docs, types, schemas, migrations, and callers consistently.

## Anti-Hallucination Rules

- Do not invent APIs, package names, config keys, commands, file paths, environment variables, database fields, or framework behavior.
- Before adding a dependency, verify that it exists, is maintained, matches the project runtime, and is compatible with current lockfiles and deployment constraints.
- Prefer official documentation, existing project usage, lockfiles, and installed package metadata over memory.
- Never claim a command, test, build, migration, benchmark, or deployment succeeded unless it was actually run or explicitly not runnable.
- If a command cannot be run, state the blocker and provide the exact verification command.
- Do not silently ignore failing tests, type errors, lint errors, security warnings, migration issues, or dependency conflicts.
- Do not replace a hard problem with a stub, mock, simplified branch, fallback constant, or broad exception handler.

## Testing and Verification

- Add or update tests that exercise the real behavior, not only mocked internals.
- Use mocks only at true external boundaries: network services, payment providers, time, filesystem, hardware, or nondeterministic systems.
- Do not over-mock internal project logic. Prefer integration tests or focused unit tests against real project components.
- For bug fixes, add a regression test that fails before the fix and passes after it.
- For refactors, prove behavior preservation with existing and focused tests.
- For performance work, measure before and after using a repeatable benchmark or profiler.
- For security-sensitive work, verify auth, authorization, input validation, secrets handling, and unsafe execution paths.
- Done requires focused tests to pass and the changed behavior to be covered.
- If broader tests are expensive, run the narrowest reliable set and document the next verification command.

## Debugging and Root-Cause Work

- Trace the failing path through entrypoint, caller, callee, state mutation, side effects, and persistence.
- Prefer structured evidence: stack traces, logs, failing assertions, data samples, query results, type errors, and profiler output.
- Reduce noisy output with `tail`, `head`, `rg`, `jq`, focused test selection, and log filters.
- Do not make speculative edits without evidence.
- If multiple root causes are possible, list hypotheses in `.agent/PLAN.md` and eliminate them with targeted checks.
- When fixing concurrency, caching, state, or lifecycle bugs, inspect initialization, invalidation, cleanup, locking, retries, and idempotency.

## Code Quality

- Keep code readable, typed where the project uses typing, and aligned with existing naming and module boundaries.
- Avoid cleverness that hides behavior. Prefer clear control flow and explicit failure handling.
- Avoid generic abstraction unless at least two real call sites need it now.
- Avoid broad `except`, silent fallback, global mutable state, hidden I/O, and implicit environment coupling.
- Maintain backward compatibility unless explicitly instructed otherwise.
- Update documentation only when it reflects real implemented behavior.

## Completion Criteria

A task is complete only when:

- The root cause or required behavior is identified.
- The implementation uses the real project stack and existing architecture.
- No fake, placeholder, mock-only, or one-case solution remains.
- Focused verification was run, or the exact blocker is documented.
- Relevant tests, types, schemas, docs, and callers are updated.
- The diff is minimal, coherent, and reviewable.
