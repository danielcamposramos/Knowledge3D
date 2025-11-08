# AI Selection Guide: Which AI for Which Task

**Key insight**: Different AI models have different strengths. Strategic selection optimizes quality, cost, and speed.

---

## The K3D Swarm (Proven in Production)

From Nov 2024 - Nov 2025, Daniel Ramos refined this AI selection strategy across 125+ development chains:

### Capability Hierarchy (Coding Quality)

```
1. Codex (VSCode) ████████████████████ Top executor
   - Huge context window
   - Direct repo write access
   - Best for: Implementation, batch file operations

2. Qwen Code (Browser) ██████████████████ Bridge tier
   - Nice context
   - Production-ready focus
   - Best for: Design + implementation hybrid

3. Claude (VSCode) ███████████████ Excellent quality
   - Moderate context (strict limits, good automatic context wrapping when it ends)
   - World-class IDE integration
   - Best for: Focused implementation, corrections, integration, comprehensive project analysis and grounding, repo based ideation

4. Qwen Max (Browser) ███████████ Good coding
   - King-level context
   - Optimization focus
   - Best for: Design, analysis, review

5. GLM/DeepSeek/Others ███████ Design & analysis
   - Massive context (GLM = KING)
   - Theory + analysis
   - Best for: Architecture, peer review
```

### Context Window Tiers

**KING tier** (Largest - holds entire development chains):
- **GLM 4.6** - Absolute largest, perfect for multi-round chains
- **Qwen Max** - On par with GLM, optimization specialist
- **Qwen Code** - Nice level + production code quality
- **DeepSeek** - Strong context, systems integration focus

**HUGE tier** (Implementation executors):
- **Codex** - Critical for seeing entire codebase

**MODERATE tier**:
- **Claude Sonnet 4.5** - Excellent but usage-limited (cost)

### Cost Optimization

**Free tiers available**:
- ChatGPT (limited daily)
- Claude (limited daily)
- Gemini (free tier)
- Grok (limited on X)
- Qwen (both - free tier)
- DeepSeek (free tier)
- GLM 4.6 (free tier)

**Paid subscriptions** (K3D uses):
- **Claude** (2×): Personal + EchoSystems AI Studios accounts (~$40/month total)
- **Codex** (2×): Personal + EchoSystems AI Studios (~$40/month total)

**Strategic cost savings**:
1. Use **browser AIs** (GLM, Qwen, DeepSeek) for design/review (king context, often cheaper if more usage is needed)
2. Use **VSCode AIs** (Codex, Claude) only for implementation (where repo access matters)
3. Result: Optimized spend - ~$80/month for 9-AI swarm vs. $200+ for all-premium

---

## Selection Matrix by Task Type

### Task: Initial Design/Architecture

**Best choice (in order of preference)**: Claude, Grok Expert, GLM, Qwen Code
**Why**: Need creativity + true github access (some AIs have limmited access to github) + broad context for exploring design space
**Avoid**: Codex (optimized for implementation, not design exploration)

**Prompt focus**:
```
- "Design the high-level architecture"
- "Explore 3 alternative approaches"
- "What are the trade-offs?"
```

**Expected output**: Conceptual design, component breakdown, alternatives

---

### Task: Peer Review / Critical Analysis

**Best choice**: Grok, GPT, GLM, DeepSeek, Qwen Max
**Why**: Need rigor + large context to see full design
**Avoid**: Fast models (may skim, miss issues)

**Prompt focus**:
```
- "Find security holes"
- "What edge cases are missing?"
- "Critique this rigorously"
```

**Expected output**: Issue list with severity, suggested fixes

---

### Task: Performance Optimization

**Best choice**: Kimi, Claude, Qwen Code
**Why**: Qwen's optimization focus, Kimi's speed demon personality
**Avoid**: General-purpose models (may not go deep on performance)

**Prompt focus**:
```
- "Optimize for <35µs latency"
- "Reduce memory usage 10×"
- "How does this scale to 1M requests/sec?"
```

**Expected output**: Optimized design, benchmarks, resource estimates

---

### Task: Implementation (VSCode with Repo Access)

**Best choice**: Codex (1st), Claude (2nd)
**Why**: Native VSCode integration, repo write access
**Avoid**: Browser AIs (no repo access - requires manual copy/paste)

**Prompt focus**:
```
- "Implement this design"
- "Create 3 files: kernel, bridge, manager"
- "Run tests after implementation"
```

**Expected output**: Code files committed to repo, test results

---

### Task: Implementation (Browser-based, Copy/Paste Workflow)

**Best choice**: Claude (1st), Codex (2nd), Qwen Code (3rd)
**Why**: Production-ready code, large context
**Avoid**: Grok Fast (optimized for speed, not code quality)

**Prompt focus**:
```
- "Implement this in production-ready [LANGUAGE]"
- "Include complete error handling"
- "Provide tests"
```

**Expected output**: Complete code block to copy into your project

---

### Task: Standards/Spec Writing (W3C)

**Best choice**: Claude, Gemini, GPT
**Why**: Formal writing quality, W3C conventions knowledge
**Avoid**: Code-specialized models (may be too technical, not formal enough)

**Prompt focus**:
```
- "Write W3C specification section"
- "Use RFC 2119 keywords (MUST/SHOULD/MAY)"
- "Include normative references"
```

**Expected output**: Formal spec text, conformance requirements, examples

---

### Task: Documentation / Tutorials

**Best choice**: Claude, Jules, Gemini, NotebookLM
**Why**: Clear explanations, user-focused writing
**Avoid**: Highly technical models (may be too terse)

**Prompt focus**:
```
- "Create tutorial for [AUDIENCE]"
- "Include step-by-step examples"
- "Explain 'why' not just 'how'"
```

**Expected output**: User-friendly docs with examples

---

### Task: Security Audit

**Best choice**: GPT, Claude, GLM, DeepSeek
**Why**: Comprehensive security knowledge, attention to detail
**Avoid**: Fast models (may miss subtle vulnerabilities)

**Prompt focus**:
```
- "Find security vulnerabilities"
- "Assume adversarial attacker"
- "Check OWASP Top 10"
```

**Expected output**: Vulnerability report, attack vectors, mitigations

---

### Task: Test Case Generation

**Best choice**: GLM, DeepSeek, Jules, Qwen (both)
**Why**: Systems thinking (DeepSeek), thorough testing mindset
**Avoid**: Design-focused models (may not think adversarially)

**Prompt focus**:
```
- "Generate test cases covering edge cases"
- "What inputs break this?"
- "Provide assertions for each case"
```

**Expected output**: Comprehensive test suite

---

### Task: Refactoring / Code Quality

**Best choice**: Claude, Codex, Jules
**Why**: Code comprehension + repo context
**Avoid**: Browser AIs without repo access (can't see full codebase)

**Prompt focus**:
```
- "Refactor for maintainability"
- "Reduce complexity"
- "Extract common patterns"
```

**Expected output**: Refactored code, diff explanation

---

## Multi-AI Combinations (Proven from K3D)

### Combination 1: Design + Review + Implement (3-AI Chain)

**Typical flow**:
1. **Grok Expert** → Initial design (creative exploration)
2. **GPT** → Peer review (rigorous critique)
3. **Codex** → Implementation (production code)

**When to use**: Standard feature development
**Time**: 30-60 minutes total
**Quality**: High (peer-reviewed by 2 AIs before code)

---

### Combination 2: Theory → Practice (GLM + Claude)

**Typical flow**:
1. **GLM** → Architecture theory, mathematical formalism (3,286 lines)
2. **GLM again** → Production code from theory (992 lines)
3. **Claude** → Implement GLM's design in repo (666 lines)

**When to use**: Complex systems requiring theoretical foundation
**Time**: Multiple hours (but parallelizable)
**Quality**: Excellent (theory-grounded implementation)

**K3D evidence**: Step 11, Round 1 + Round 2

---

### Combination 3: Round-Robin Enhancement (5-AI Chain)

**Typical flow**:
1. **Grok** → Base design
2. **Qwen Max** → Add caching/optimization
3. **Kimi** → Add performance tuning
4. **DeepSeek** → Add integration logic
5. **GLM** → Synthesize + add theory

**When to use**: Major features requiring multiple perspectives
**Time**: 1-2 hours (sequential)
**Quality**: Exceptional (5 peer reviews)

**K3D evidence**: Step 11, Round 1 (5,642 lines from 5 AIs)

---

### Combination 4: Parallel Review (2 AIs Review Same Design)

**Typical flow**:
1. **AI #1** (Grok) → Design
2. **AI #2** (GPT-4) → Review (independently)
3. **AI #3** (Claude) → Review (independently, doesn't see AI #2's review)
4. **Human** → Synthesize both reviews
5. **AI #4** (Codex) → Implement addressing both reviews

**When to use**: Critical features, security-sensitive code
**Time**: +20 minutes (parallel reviews)
**Quality**: Maximum (two independent critical perspectives)

---

## Access Methods & Trade-offs

### Browser-Based AIs

**Advantages**:
- ✅ Accessible to everyone (no special software)
- ✅ Often cheaper than IDE-integrated
- ✅ King-level context windows (GLM, Qwen, DeepSeek)
- ✅ Fresh perspectives (no repo bias)

**Disadvantages**:
- ❌ No direct repo access (manual copy/paste)
- ❌ Can't run tests automatically
- ❌ Can't see full codebase structure

**Best for**: Design, review, enhancement, specification writing

**Examples**: Grok (grok.com), Qwen (qwen.ai), GLM (chatglm.cn), DeepSeek (deepseek.com), Claude (claude.ai), ChatGPT (openai.com)

---

### VSCode-Integrated AIs

**Advantages**:
- ✅ Direct repo access (reads/writes files)
- ✅ Can run tests + commit changes
- ✅ Sees full codebase structure
- ✅ World-class IDE integration

**Disadvantages**:
- ❌ Requires VSCode setup
- ❌ Often have usage limits (cost)
- ❌ May have "repo bias" (follows existing patterns, less creative)

**Best for**: Implementation, refactoring, testing, integration

**Examples**: Claude Code (Sonnet 4.5), GitHub Copilot (Codex)

---

## Special-Purpose AIs

### NotebookLM (Google)

**Strengths**:
- Excellent at synthesizing multiple documents
- Creates summaries, FAQs, study guides
- Can generate audio discussions

**Weaknesses**:
- Not code-focused
- Can't implement designs

**Best for**: Documentation, briefing generation, meta-analysis

**K3D use**: Created K3D Briefing initial draft

---

### Jules (code review specialist)

**Strengths**:
- Deep code review focus
- Comments, Q&A, critique

**Weaknesses**:
- Not design-focused

**Best for**: Code review, technical Q&A

**K3D use**: Peer review of implementations

---

## Selection Decision Tree

```
START: What's your task?

├─ "I need initial design/architecture"
│  ├─ Need creative exploration → Grok Expert, GLM
│  ├─ Need production-ready code sketch → Qwen Code
│  └─ Need theoretical foundation → GLM
│
├─ "I need peer review/critique"
│  ├─ Security-focused → GPT-4, Claude
│  ├─ Performance-focused → Qwen Max, Kimi
│  └─ Comprehensive → GLM (can hold entire design in context)
│
├─ "I need implementation"
│  ├─ Have VSCode + repo access → Codex (1st), Claude (2nd)
│  ├─ Browser-based copy/paste → Qwen Code (1st), Claude/Codex (2nd)
│  └─ Need ultra-production-ready → Qwen Code
│
├─ "I need specification/standards"
│  └─ W3C/formal specs → GPT-4, Claude, Gemini
│
├─ "I need documentation"
│  └─ User-facing → Claude, Gemini, NotebookLM
│
└─ "I need [other task]"
   └─ See task-specific recommendations above
```

---

## Cost-Optimized Strategy (W3C Recommended)

**If budget-constrained**:

1. **Free tiers only** (totally feasible):
   - ChatGPT (design, review, spec writing)
   - Claude (implementation, documentation)
   - Gemini (review, documentation)
   - **Cost**: $0/month
   - **Limit**: Daily usage caps, but sufficient for most spec work

2. **One paid subscription** ($20/month):
   - **ChatGPT Plus** or **Claude Pro**
   - Unlimited usage, higher quality
   - **Use for**: All stages (one AI does design, review, implement, document)
   - **Trade-off**: Lose multi-perspective benefit, but still works

3. **Two paid subscriptions** ($40/month - K3D approach):
   - **Claude Pro** ($20) for implementation/docs
   - **ChatGPT Plus** ($20) for design/review
   - **Use free**: Gemini for additional reviews
   - **Result**: Multi-vibe benefits at minimal cost

4. **Full K3D swarm** (~$80/month):
   - Claude × 2 (personal + org account)
   - Codex × 2 (personal + org account)
   - + Free browser AIs (Qwen, GLM, DeepSeek, Grok)
   - **Result**: 9-AI capability, maximum perspectives

**W3C CG recommendation**: Start with Option 1 (free), upgrade to Option 2 ($20) if productive.

---

## Common Selection Mistakes

### Mistake 1: Using Same AI for All Stages

**Problem**: Lose multi-perspective benefit, AI may repeat own biases

**Solution**: Minimum 3 different AIs (design, review, implement)

---

### Mistake 2: Using Free Fast Models for Critical Review

**Problem**: Fast models may skim, miss issues

**Solution**: Use GPT-4, Claude, or GLM for peer review (worth the cost/time)

---

### Mistake 3: Using Browser AI When Repo Access Needed

**Problem**: Manual copy/paste → errors, inefficiency

**Solution**: Use Codex/Claude in VSCode for repo operations

---

### Mistake 4: Using Code-Specialized AI for Documentation

**Problem**: Documentation too technical, not user-friendly

**Solution**: Use Claude, Gemini, or NotebookLM for user-facing docs

---

### Mistake 5: Ignoring Context Window Limits

**Problem**: AI loses track of previous work mid-conversation

**Solution**:
- Use king-context AIs (GLM, Qwen, DeepSeek) for long chains
- Or break into smaller chains
- Or manually refresh context (see [Context Management](./06_context_management.md))

---

## Next Steps

**Try the selection strategy**: [Getting Started Guide](./01_getting_started.md) uses Grok → Claude → Codex chain

**Understand context limits**: [Context Management](./06_context_management.md) explains when to switch AIs

**See it in action**: [K3D Step 11 Case Study](./11_case_study_step11.md) shows 6-AI selection in practice

**Apply to W3C**: [W3C Standards Application](./08_w3c_standards.md) recommends AI selection for spec work

---

**Remember**: **Strategic AI selection** is one of the three pillars of multi-vibe (along with partnership invocation and copy-paste discipline). Choose wisely!
