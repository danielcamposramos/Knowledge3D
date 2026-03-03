# WebML Open Issues Analysis

Date: 2026-03-03
Repository: `https://github.com/webmachinelearning/proposals`
Scope analyzed: all open proposal issues in the repository

## 1. Repository Snapshot

- Total proposal issues (non-PR): 16
- Open issues: 15
- Closed issues: 1
- Repo description: "Proposals for future work"
- Proposal mechanism: GitHub Issues
- Template used: `.github/ISSUE_TEMPLATE/new-proposal.md`

Template sections required by WebML proposals repo:
1. Proposal name
2. Short description
3. Example use cases
4. A rough idea or two about implementation

## 2. Open Issues (All 15)

| # | Title | Author | Opened (UTC) | Comments | Last Update (UTC) | Current Status Signal | Related Links (key) |
|---|---|---|---|---:|---|---|---|
| 1 | Data processing proposal | @WenheLI | 2020-11-11 | 11 | 2021-01-22 | Legacy; discussed in calls; no closure | WebML meetings agendas |
| 2 | Operation-specific APIs | @jbingham | 2020-12-15 | 10 | 2021-03-22 | Legacy; requirements partially absorbed into WebNN work; kept open | WebNN PR #149, #154, issue #157 |
| 3 | Supporting JAX-inspired WebML frameworks/libraries | @josephrocca | 2021-10-15 | 1 | 2021-10-16 | Early discussion only; dormant | WebNN issue #218 |
| 4 | Add use case on Content Filtering in WebNN specifications | @humeranoor | 2022-08-26 | 4 | 2022-09-14 | Redirect pressure toward adjacent scope (WebExtensions); no formal close | WebNN issue #236, PR #253 |
| 5 | Hybrid AI Exploration | @mmccool | 2024-02-29 | 14 | 2025-08-08 | Incubated and split to dedicated repo; issue intentionally left open | `webmachinelearning/hybrid-ai` |
| 6 | Support Adapters to modify existing MLGraph objects | @mmccool | 2025-01-31 | 0 | 2025-02-04 | Seed proposal; no discussion yet | hybrid-ai adapters proposal |
| 7 | Proofreader API | @domenic | 2025-03-17 | 2 | 2025-08-08 | Scoped discussion + pointer to dedicated explainer repo | `webmachinelearning/proofreader-api` |
| 8 | Prompt API: local use cases for RAG/Agentic RAG | @zolkis | 2025-03-24 | 10 | 2025-04-30 | Active conceptual discussion; cross-reference to Prompt API track | `webmachinelearning/prompt-api` |
| 9 | Local Inference Web extension | @dontcallmedom | 2025-04-29 | 0 | 2025-04-29 | Seed proposal; no thread activity yet | Mozilla extension blog, Searchfox link |
| 10 | Web AI for Time Series | @AdamSobieski | 2025-05-07 | 2 | 2025-05-08 | Initial concept discussion only | N/A |
| 11 | Fact-checking API | @AdamSobieski | 2025-05-28 | 2 | 2026-02-11 | Ongoing caution/safety debate; no incubator repo announced | OSF preprint link |
| 12 | WebMCP API (initially Web Model Context API) | @sushraja-msft | 2025-07-30 | 12 | 2025-08-19 | Strong incubation path; dedicated repos created and contributors onboarded | `webmachinelearning/webmodelcontext`, `webmachinelearning/webmcp` |
| 14 | MCP-UI & web spec enhancements | @anssiko | 2025-10-08 | 0 | 2025-10-08 | New proposal seed; no issue-thread discussion yet | MCP-UI references |
| 15 | Dynamic AI Offloading Protocol (DAOP) | @jonathanding | 2026-01-09 | 4 | 2026-02-06 | Active; explicit WG resolution to create explainer and prototype | 2026-01-15 WG resolution/minutes |
| 16 | Standardize a WebNN Graph DSL and Portable File Format | @tarekziade | 2026-02-13 | 2 | 2026-02-16 | Active and converging; collaboration discussion with related syntax effort | `rustnn/webnn-graph` |

## 3. Per-Issue Notes (Status and Signals)

### #1 Data processing proposal
- Maintainer engagement happened early (agenda placement and call invitations).
- No recent activity since 2021.
- Status signal: dormant legacy proposal.

### #2 Operation-specific APIs
- Significant technical discussion with WebNN editors.
- Thread references concrete WebNN artifacts (PRs/issues) and states requirements being addressed in WebNN.
- Status signal: partially absorbed by primary WebNN spec track.

### #3 Supporting JAX-inspired frameworks
- Single maintainer acknowledgment comment.
- No follow-up comments, no explainer/repo spin-out.
- Status signal: dormant exploratory idea.

### #4 Content Filtering use case
- Discussion indicates scope tension (WebNN use case text vs broader extension APIs).
- Maintainer explanation points to case-driven scope discipline.
- Status signal: unresolved scope alignment.

### #5 Hybrid AI Exploration
- Highest discussion density among open proposals.
- Evidence of formal WG discussions and project-team summaries.
- Dedicated repo created for continued structured work.
- Status signal: active incubation via repo migration.

### #6 Adapters for MLGraph
- No comments; proposal points to external hybrid-ai document.
- Status signal: waiting for championing and discussion.

### #7 Proofreader API
- Discussed in meetings; linked to dedicated proofreader explainer work.
- Status signal: scoped into adjacent/dedicated API work.

### #8 Prompt API local RAG use cases
- Active comments and cross-linking to Prompt API goals and TAG/explainer process.
- Status signal: still framing requirements and governance.

### #9 Local Inference Web extension
- No comments since opening.
- Proposal intentionally targets rapid prototyping and extension layer.
- Status signal: untriaged seed.

### #10 Web AI for Time Series
- Concept-level framing, low comment activity.
- Status signal: early-stage concept, no standardization path yet.

### #11 Fact-checking API
- Contains direct pushback on API-level risk and model governance.
- Follow-up reframes as assistant-mediated UX patterns.
- Status signal: contested and policy-sensitive.

### #12 WebMCP API
- Strongest process maturity signal: maintainer confirms support direction, creates dedicated repo, moves contributors.
- Follow-up acknowledges second dedicated repo (`webmcp`) and community onboarding.
- Status signal: successful incubation pattern.

### #14 MCP-UI enhancements
- Seed issue opened by maintainer with references and collaboration intent.
- No comments yet.
- Status signal: new intake item.

### #15 DAOP
- Contains explicit WG resolution: create explainer and begin prototyping.
- Discussion balances privacy/feasibility constraints vs implementation approach.
- Status signal: active with concrete next step.

### #16 WebNN Graph DSL
- Early but high-quality technical exchange.
- Signals possible convergence with adjacent syntax efforts (WebNNM mention in comments).
- Status signal: promising active proposal.

## 4. Open-Issue Landscape Summary

- The repository is low-volume but curated: 15 open proposals over multiple years.
- Maturity pattern is clear: issues that get maintainer orchestration move to dedicated repos/explainers.
- No formal labels are used; process signals are in comments, minutes links, and repo spin-outs.
- The most active themes are:
  - Hybrid/offloading and model placement (#5, #15)
  - Agent/tool protocols (#12, #14)
  - Model representation and tooling interoperability (#16)
  - User-facing AI APIs with policy constraints (#7, #8, #11)
