# Privacy scrub report

Date: 2026-10-27

Scope: every file under this release folder (including hidden/ignored files, binary arrays treated as text, this report and `docs/index.html`), plus the separate content-only artifact. All 26 S4 patterns are retained; an exact full-author-credit pattern is added. Scanning uses case-insensitive ripgrep PCRE2. Counts are match occurrences, not lines. Historical files in S4/_superseded, tools and local check workspaces are outside the release scope.

Only README author headers and copyright lines, NOTICE copyright, CITATION author fields and map footer credits may match the Latin author patterns. Source footer strings in `code/page.js` are the same permitted credit. Every hit is listed below by file, line and one-based byte column, without repeating the name outside its permitted locations. Required upstream idea credit remains; no upstream code was used. Individual human review marks, notes and reviewer quotations are excluded. Original scientific measurements and classifier behavior are preserved.

| Pattern category | Checked PCRE2 pattern | Final count |
|---|---|---:|
| user-home-path | `[/]Users[/]` | 0 |
| unix-home-path | `[/]home[/]` | 0 |
| volume-path | `[/]Volumes[/]` | 0 |
| personal-machine-name | `[j]ihosmacbookpro` | 0 |
| alternate-handle | `sa[l]ix` | 0 |
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

The token and full-credit patterns overlap; both match lists are shown. The citation given-name field matches only the token pattern.

| Category | File | Line | Byte column | Allowed credit |
|---|---|---:|---:|---|
| author-latin-name | `CITATION.cff` | 6 | 19 | yes |
| author-latin-name | `NOTICE` | 1 | 21 | yes |
| author-latin-name | `README.md` | 3 | 16 | yes |
| author-latin-name | `README.md` | 5 | 21 | yes |
| author-latin-name | `README_ko.md` | 3 | 19 | yes |
| author-latin-name | `README_ko.md` | 5 | 21 | yes |
| author-latin-name | `code/page.js` | 32 | 478 | yes |
| author-latin-name | `code/page.js` | 45 | 2241 | yes |
| author-latin-name | `docs/index.html` | 47 | 478 | yes |
| author-latin-name | `docs/index.html` | 60 | 2241 | yes |
| author-latin-name | `map/artifact_body.html` | 45 | 478 | yes |
| author-latin-name | `map/artifact_body.html` | 58 | 2241 | yes |
| author-full-credit | `NOTICE` | 1 | 16 | yes |
| author-full-credit | `README.md` | 3 | 11 | yes |
| author-full-credit | `README.md` | 5 | 16 | yes |
| author-full-credit | `README_ko.md` | 3 | 14 | yes |
| author-full-credit | `README_ko.md` | 5 | 16 | yes |
| author-full-credit | `code/page.js` | 32 | 473 | yes |
| author-full-credit | `code/page.js` | 45 | 2236 | yes |
| author-full-credit | `docs/index.html` | 47 | 473 | yes |
| author-full-credit | `docs/index.html` | 60 | 2236 | yes |
| author-full-credit | `map/artifact_body.html` | 45 | 473 | yes |
| author-full-credit | `map/artifact_body.html` | 58 | 2236 | yes |

Every other pattern must be zero. These regexes test specified identifiers and conventional formats, not every conceivable personal fact. Stage/model names and original scientific hashes are intentional. Synthetic scenarios are authored research fixtures.
