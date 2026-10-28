# Privacy scrub report

Date: 2026-10-28

Scope: every file in the staged public tree, including hidden/ignored files, binary arrays scanned as text, this report and `docs/index.html`. The page builder's `artifact_body.html` (the same page body) was scanned too, then kept out of the published tree. All 27 round-1 patterns (the 26 S4 patterns plus exact author credit) are retained. Case-insensitive ripgrep PCRE2 counts match occurrences, not lines. Work-folder check logs, decision drafts and `_superseded/` archives are outside the staged tree.

Author and footer credits retain the round-1 exact-location allowance. The two inherited README Pages URL lines have one narrow public-link allowance: the round-1 report predates those link additions. All occurrences are listed below, including overlapping author patterns. No other occurrence is permitted. No identifier is reproduced here outside an encoded pattern or a file/line location.

Round-2 source extraction uses a field allowlist: aggregate metrics/counts only; no cases, replies, annotation text, raw context, token IDs, activations, adapter/package files, checkpoints or app source. Character labels are A–D. RULE copies neutralize creator/character identities, package/environment references and follow-up private scenario IDs/model manifest; source and distributed hashes are separate. No round-2 individual human review marks or words are included. Inherited authorized round-1 illustrative replies, aggregate review counts and steering vectors retain their existing scope. Only synthetic round-2 fixtures were used; no own-chat contents were encountered in the reviewed inputs.

| Pattern category | Checked PCRE2 pattern | Count |
|---|---|---:|
| user-home-path | `[/]Users[/]` | 0 |
| unix-home-path | `[/]home[/]` | 0 |
| volume-path | `[/]Volumes[/]` | 0 |
| personal-machine-name | `[j]ihosmacbookpro` | 0 |
| alternate-handle | `sa[l]ix` | 2 |
| volume-label | `J[I]HO[_]SSD` | 0 |
| compute-host | `\bo[m]en\b` | 0 |
| private-ip-prefix | `192[.]168[.]` | 0 |
| email | `[A-Z0-9._%+\-]+@[A-Z0-9.\-]+[.][A-Z]{2,}` | 0 |
| hub-bucket-or-repository | `(?:h[f]://|https?://(?:www[.])?(?:huggingface[.]co|hf[.]co)/|H[F][_]BUCKET|H[F][_]REPO|hf[_\-](?:bucket|repo))` | 0 |
| uuid-device-identifier | `\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b` | 0 |
| hardware-address | `\b(?:[0-9a-f]{2}:){5}[0-9a-f]{2}\b` | 0 |
| local-hostname | `\b[a-z0-9-]+[.]local\b` | 0 |
| application-switch | `p[v]a[.]` | 0 |
| application-agent-identifier | `\bbot[_\-]?(?:id|[0-9a-z]{12,})\b` | 0 |
| application-internal-class | `Visible[R]eplyCheck|Visible[W]hoIsWho` | 0 |
| character-01 | `\x{c11c}\x{d558}` | 0 |
| character-02 | `\x{c608}\x{c9c4}` | 0 |
| character-03 | `\x{b9ac}\x{c5d4}` | 0 |
| character-04 | `\x{c11c}\x{b9b0}` | 0 |
| character-05 | `\x{d558}\x{b9b0}` | 0 |
| character-06 | `\x{c740}\x{d638}` | 0 |
| character-07 | `\x{bbf8}\x{b098}` | 0 |
| character-08 | `\x{b3c4}\x{c724}` | 0 |
| author-korean-name | `\x{c9c0}\x{d638}` | 0 |
| author-latin-name | `\bji[h]o\b` | 12 |
| author-full-credit | `\bRyou Ji[h]o\b` | 11 |

## Every hit

| Category | File | Line | Byte column | Allowed credit / public link |
|---|---|---:|---:|---|
| alternate-handle | `README.md` | 28 | 36 | yes |
| alternate-handle | `README_ko.md` | 24 | 41 | yes |
| author-latin-name | `CITATION.cff` | 6 | 19 | yes |
| author-latin-name | `NOTICE` | 1 | 21 | yes |
| author-latin-name | `README.md` | 3 | 16 | yes |
| author-latin-name | `README.md` | 5 | 21 | yes |
| author-latin-name | `README_ko.md` | 3 | 19 | yes |
| author-latin-name | `README_ko.md` | 5 | 21 | yes |
| author-latin-name | `code/page.js` | 40 | 478 | yes |
| author-latin-name | `code/page.js` | 61 | 2260 | yes |
| author-latin-name | `docs/index.html` | 55 | 478 | yes |
| author-latin-name | `docs/index.html` | 76 | 2260 | yes |
| author-full-credit | `NOTICE` | 1 | 16 | yes |
| author-full-credit | `README.md` | 3 | 11 | yes |
| author-full-credit | `README.md` | 5 | 16 | yes |
| author-full-credit | `README_ko.md` | 3 | 14 | yes |
| author-full-credit | `README_ko.md` | 5 | 16 | yes |
| author-full-credit | `code/page.js` | 40 | 473 | yes |
| author-full-credit | `code/page.js` | 61 | 2255 | yes |
| author-full-credit | `docs/index.html` | 55 | 473 | yes |
| author-full-credit | `docs/index.html` | 76 | 2255 | yes |

All occurrences outside these exact credit/public-link allowances must be zero. The scan was rerun after this report and the complete checksum manifest were written. Regex checks cover the specified identifiers and formats, not every conceivable personal fact. No inference or new semantic annotation was performed while staging.
