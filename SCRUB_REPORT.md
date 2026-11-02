# Privacy scrub report

Date: 2026-11-03

All 27 round-1 identifier patterns were run over the whole staged tree, including compressed bytes. Every JSON gzip shard in the single Pages data copy was also decoded and scanned. Numeric fixed-width card/catalog files contain only schema-validated numbers and IDs. No matched text is printed or logged. The final read-only seal check includes this report and both checksum manifests.

Allowed scopes: inherited author credits, exact public repository/Issues links, exact open-dataset attribution links, and public UUID-shaped row IDs within pointer records only. Corpus snippets have no identifier exception. Inherited published examples retain their prior scope; no new private chat, judge context/rationale, app, adapter or character materials were extracted.

| Identifier category | Raw count | Decoded JSON count |
|---|---:|---:|
| user-home-path | 0 | 0 |
| unix-home-path | 0 | 0 |
| volume-path | 0 | 0 |
| personal-machine-name | 0 | 0 |
| alternate-handle | 4 | 0 |
| volume-label | 0 | 0 |
| Host-name pattern | 0 | 0 |
| private-ip-prefix | 0 | 0 |
| email | 0 | 0 |
| hub-bucket-or-repository | 3 | 0 |
| uuid-device-identifier | 0 | 0 |
| hardware-address | 0 | 0 |
| local-hostname | 0 | 0 |
| application-switch | 0 | 0 |
| application-agent-identifier | 0 | 0 |
| application-internal-class | 0 | 0 |
| Project-private name list | 0 | 0 |
| author-latin-name | 10 | 0 |
| author-full-credit | 9 | 0 |

## Open-source snippets

All 20 top-context ranks are eligible; N=20, with no budget cap. Snippets use 15 tokens before + firing token + 16 after, clipped to at most 32 content tokens. Each clipped window is stored once. No web context text is distributed.

The exact 200,000,000-byte token stream passed its frozen SHA256 check; 91,836 candidate/boundary chunks passed their individual token hashes. Whole-chunk M0 patterns, M3 boundary joins/first 32-character digit fragments, and snippet-level patterns run before publication. Boundary checks use adjacent chunks within the same source document; an unavailable previous chunk would be excluded. A fixed Korean/English lexical list conceals matching snippets until a click. Pattern filtering does not guarantee de-identification.

| Open source | Candidate chunks | Unique snippets | Neuron-context references | Hidden references | Covered tokens | Dev coverage | Full fixed-slice coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| songys/Chatbot_data | 6,957 | 8,996 | 12,680 | 8 | 129,093 | 60.546% | 59.900% |
| mkd-chanwoo/keural-conversation-ko | 48,873 | 250,602 | 285,496 | 365 | 3,607,514 | 72.896% | 72.150% |
| allenai/soda | 29,755 | 486,209 | 594,320 | 516 | 4,506,324 | 91.073% | 90.126% |
| deepmind/pg19 | 5,871 | 150,756 | 199,336 | 2 | 1,380,098 | 92.007% | 92.007% |

References count neuron/context slots; identical clipped windows are stored once. Coverage is the union of token intervals per corpus chunk. Dev and full fixed-source-slice denominators are distinct; no heldout text is published.

| Drop category | Chunks | Context references |
|---|---:|---:|
| Host-name pattern | 4 | 124 |
| Project-private name list | 67 | 527 |
| account | 0 | 0 |
| address | 97 | 1,743 |
| boundary_address | 79 | 1,025 |
| boundary_phone | 64 | 931 |
| boundary_to_next_chunk | 1 | 125 |
| email | 0 | 0 |
| head_digit_fragment | 3 | 256 |
| missing_previous_chunk | 0 | 0 |
| phone | 66 | 1,030 |
| resident_registration | 0 | 0 |

The project-private name row is a distinct union, counted once per chunk/reference across its patterns. Other categories may overlap. There are 229 distinct rejected chunks and 3,618 rejected references; additional snippet-only rejections are zero. The 891 hidden references are retained behind a click, not deleted.

## Status

PASS: all current-stage identifier matches are within the explicit public scopes.

## M5i Korean number particles

The Korean overview now chooses its number particle from the last spoken Sino-Korean syllable. All layer labels L0–L34 and nine neuron/unit cases passed. The guide also uses the correct topic particle for L0. The English string dictionary is byte-equivalent under JSON serialization. All 7,071 compressed data files remain unchanged; no excerpt, source, privacy or concealment policy changed.

## Size budget

| Item | Bytes | Limit |
|---|---:|---:|
| New files | 238964254 | 400,000,000 |
| New and modified files | 242889140 | 400,000,000 |
| Entire stage including checksums | 247501474 | — |
| Initial assets and metadata | 102053 | 2,000,000 |
| Overview ready | 1092719 | 2,000,000 |
| Cold layer grid, maximum | 1193461 | 2,000,000 |
| Cold 50-row list, maximum | 1684608 | 2,000,000 |
| Direct card, maximum | 1386416 | 2,000,000 |
| Largest map data file | 1169766 | 25,000,000 |
| Largest inflated map shard | 3523255 | 25,000,000 |

All data and snippet counts, coverage and filter drops above are unchanged from M5h. Research values and the selected examples are unchanged. File sizes count bytes, not disk allocation.
