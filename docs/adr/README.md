# Architecture decision records

This directory records the architectural decisions of SedStat: what was
decided, why, and what follows from it. One file per decision, in the format
of Michael Nygard's
[Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

## When to write one

A decision needs a record when it constrains more than one module or is
expensive to reverse.

## Writing a record

1. Copy [`template.md`](template.md) to `NNNN-short-title.md`, taking the next
   free number. Numbers are never reused.
2. Fill in context, decision and consequences. Keep it short; a record that
   needs more than a page is usually two decisions.
3. Add a row to the index below.

Do not edit a record once it is `accepted`. If a decision changes, write a
new record, set the old one to `superseded by NNNN` and link both ways.

## Status values

| Status | Meaning |
|---|---|
| `proposed` | Under discussion, not yet binding. |
| `accepted` | Binding for all new code. |
| `superseded by NNNN` | Replaced by a later record. |
| `deprecated` | No longer applies, without a replacement. |

## Index

| No. | Title | Status |
|---|---|---|
