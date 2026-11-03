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

## M5k failed-load recovery

The shared JSON/gzip cache deduplicates pending work and retains successful loads. It removes an entry when fetch, JSON parsing or gzip decompression rejects, checking promise identity before removal. Fetch retries once after 250 ms for a rejected network fetch or a 5xx response; 404 and other client errors do not retry automatically. The browse metadata/overview promise wrappers also forget rejected loads.

Overview, grid, list and card failures now offer a short Korean/English message and a Try again button inside the affected view. Card evidence entries can retry their failed data too. Rendering keeps its generation/request guards, and old retry buttons cannot revive a view after navigation. Startup reads restored filters once and avoids a second automatic render of a missing overview shard.

The local fetch-stub matrix covers Korean/English at modeled desktop/375 px widths: eight automatic recoveries, seven manual recoveries, four explicit view-failure cases, and five delayed-request/stale-button races. JSON/gzip decode failures evict their entries, a 404 on one layer does not block another, and cached successes remain reusable. M5j filtering, restore, sort, grid tap, cards, particles and offline/privacy/budget checks were rerun. These are existing JS/DOM model checks, not a fresh real-browser measurement. All 7,071 compressed data files and privacy/snippet policies remain unchanged. No corpus or snippet text appears in test output.

## Size budget

| Item | Bytes | Limit |
|---|---:|---:|
| New files | 238994718 | 400,000,000 |
| New and modified files | 242920774 | 400,000,000 |
| Entire stage including checksums | 247533108 | — |
| Initial assets and metadata | 108535 | 2,000,000 |
| Overview ready | 1099201 | 2,000,000 |
| Cold layer grid, maximum | 1199943 | 2,000,000 |
| Cold 50-row list, maximum | 1691090 | 2,000,000 |
| Direct card, maximum | 1392898 | 2,000,000 |
| Largest map data file | 1169766 | 25,000,000 |
| Largest inflated map shard | 3523255 | 25,000,000 |

Data, coverage, filtering and research values are unchanged. File sizes count bytes, not disk allocation.

## Round 4 F2 / 4라운드 F2

All 27 inherited identifier patterns scan the full stage and every decoded JSON gzip shard, including Round 4. Unexpected matches: 0. Snippet entries checked: 1060469. No matched text is printed or logged. Exact public attribution/row-ID exceptions remain the M5 exceptions; snippets have no identity exception. Feature candidates use the same whole-chunk, boundary and window filters and lexical concealment policy. / 전체 스테이지와 압축 해제한 모든 JSON에 기존 27개 패턴을 검사했습니다. 예상 밖 일치 0개이며 문장 내용은 출력하지 않았습니다. 특징 문장에도 같은 개인정보·문서 경계·가림 규칙을 적용합니다.
