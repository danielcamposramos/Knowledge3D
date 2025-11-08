# Context Management: Handling AI Memory Limits

**The challenge**: AI context windows have limits. Long development chains can exceed them.

**The solution**: Strategic context assembly and injection (Daniel's "message board software" role)

---

## Understanding Context Windows

### What is a Context Window?

**Context window** = How much text an AI can "remember" in a single conversation.

**Analogy**: Like human short-term memory, but measured in tokens (roughly words).

### Context Window Tiers (from K3D Swarm)

| AI Model | Context Tier | Approximate Size | Practical Limit |
|----------|--------------|------------------|-----------------|
| **GLM 4.6** | **KING** | 128K+ tokens | ~100,000 words |
| **Qwen Max** | **KING** | 128K+ tokens | ~100,000 words |
| **Qwen Code** | **KING** | 128K+ tokens | ~100,000 words |
| **DeepSeek** | **Strong** | 64K+ tokens | ~50,000 words |
| **Codex** | **Huge** | 100K+ tokens | ~80,000 words |
| **GPT-4** | **Large** | 32K-128K tokens | ~25,000-100,000 words |
| **Claude Sonnet 4.5** | **Moderate** | 200K tokens | ~150,000 words (but usage limited) |
| **Gemini** | **Massive** | 1M+ tokens | ~800,000 words |

**Note**: Practical limits are lower than theoretical (AI performance degrades with very long contexts).

---

## When Context Windows Become a Problem

### Scenario 1: Long Development Chain

**Problem**: Step 11 had 9,090 lines across 6 AIs in 2 rounds.

**Math**:
- 9,090 lines × ~50 words/line = ~450,000 words
- Most AIs can't hold this in one conversation

**Daniel's solution**: Break into rounds
- **Round 1**: 5 AIs (5,642 lines) → Stayed within context for most AIs
- **Round 2**: Started fresh conversation, pasted summary of Round 1 + new task
- Result: Each AI never exceeded its context window

---

### Scenario 2: Mid-Conversation Context Loss

**Problem**: AI partner's context window fills up during active development.

**Symptom**: AI loses track of previous decisions
- "Wait, what architecture did we decide on?"
- Contradicts earlier decisions
- Asks questions already answered
- Regenerates code instead of building on previous version

**Example from K3D** (Step 11):
- GLM was working on Round 2 implementation
- Context window hit limit mid-session
- Started suggesting approaches incompatible with Round 1 design
- Daniel noticed: "This contradicts what you designed in Round 1"

**Daniel's solution**: Context assembly and re-injection
1. Identified what GLM "forgot" (Round 1 design decisions)
2. Assembled relevant excerpts (not everything - that would overflow again!)
3. Pasted: Briefing + Round 1 key decisions + current task + "Continue from here"
4. GLM caught up, continued coherently

---

### Scenario 3: Multiple Rounds with Different AIs

**Problem**: AI #6 joining chain has no context from AI #1-5's work.

**Solution**: Selective context pasting

**Don't do this** (overflow):
```
Paste entire work from AI #1 (2,000 lines)
+ entire work from AI #2 (1,500 lines)
+ entire work from AI #3 (2,500 lines)
+ entire work from AI #4 (1,800 lines)
+ entire work from AI #5 (2,200 lines)
= 10,000 lines → Context overflow!
```

**Do this** (selective):
```
Paste:
- Briefing (200 lines)
- AI #1's architecture summary (100 lines)
- AI #2's key critique points (50 lines)
- AI #3's final design (300 lines)
- AI #4's validation results (50 lines)
- AI #5's implementation approach (200 lines)
- Current task for AI #6 (50 lines)
= 950 lines → Fits comfortably, AI #6 has sufficient context
```

---

## Daniel's Context Management Techniques

### Technique 1: The "STEP File" Approach

**What**: Create cumulative development chain file, organize by rounds.

**Structure**:
```
# STEP File: [Feature Name]

## Briefing
[K3D Briefing + task-specific context]

## Round 1: Foundation
### AI #1 (Grok) - Design
[Full output]

### AI #2 (Qwen) - Cache
[Full output]

### AI #3 (Kimi) - Performance
[Full output]

### AI #4 (DeepSeek) - Integration
[Full output]

### AI #5 (GLM) - Theory
[Full output]

## Round 1 Summary (for Round 2 context)
[Daniel's 200-line summary of key decisions]

## Round 2: Production
[Repeat structure]
```

**Why this works**:
- All outputs preserved (nothing lost)
- Summaries created at round boundaries (for later reference)
- Can selectively paste sections to next AI

**K3D evidence**: 125+ STEP files in TEMP/ folder

---

### Technique 2: Citing Previous Context

**What**: When pasting previous work, cite it explicitly.

**Example**:
```
Context: In Round 1, we established these design decisions:

1. Architecture: Sovereign GPU-only (no CPU fallbacks)
   [Source: AI #1 - Grok, Round 1, lines 45-67]

2. Caching: LRU with max 1000 shapes
   [Source: AI #2 - Qwen, Round 1, lines 112-134]

3. Performance target: <35µs inference
   [Source: AI #3 - Kimi, Round 1, lines 201-223]

Your task (AI #6): Implement the testing framework for this design.
```

**Why this works**:
- AI knows where decisions came from (builds trust)
- Citations help AI prioritize (recent > old, design > comment)
- Reduces confusion about "who said what"

**K3D evidence**: STEP files frequently cite "Grok suggested X in Round 1", "GLM's theory from lines Y-Z"

---

### Technique 3: The "Message Board" Analogy

**What**: Like threading an email chain, Daniel quotes relevant previous messages.

**Email threading**:
```
> AI #1 wrote:
> > Design should use JWT tokens with 15-minute expiry

> AI #2 responded:
> > 15 minutes is too short for mobile users (network interruptions)
> > Suggest: 1 hour expiry + refresh token

AI #3: I'm implementing authentication. Which expiry should I use?
```

**Multi-vibe equivalent**:
```
Context from previous partners:

AI #1 (Grok) originally proposed:
> JWT tokens with 15-minute expiry

AI #2 (Claude) critiqued:
> 15 minutes too short for mobile users
> Suggested: 1 hour expiry + refresh token

Your task (AI #3 - Codex): Implement authentication.
Which approach do you recommend and why?
```

**Why this works**:
- AI sees the conversation history (understands context)
- AI can weigh both perspectives
- AI proposes synthesis (1-hour expiry accepted)

---

### Technique 4: Incremental Context Building

**What**: Start small, add context incrementally as needed.

**Phase 1**: Minimal context (test if AI needs more)
```
Task: Implement user authentication
Constraint: Must use JWT tokens
```

**AI response**: "Should I use symmetric or asymmetric signing?"

**Phase 2**: Add relevant context (answer AI's question)
```
Context: We're using asymmetric (RS256) for token verification
across multiple services.

[Original task + constraint]
```

**AI response**: "Understood. Should I implement refresh tokens?"

**Phase 3**: Add more context
```
Context: Yes, refresh tokens required. Design from AI #2:
[Paste AI #2's refresh token design]

[Original task + previous context]
```

**Why this works**:
- Avoids front-loading too much context (overwhelming)
- Responds to AI's actual information needs (just-in-time context)
- Keeps context window usage minimal

---

## Strategies for Different Context Window Sizes

### Strategy for KING Context AIs (GLM, Qwen Max, Qwen Code)

**Advantage**: Can hold massive chains (~100,000 words)

**Use for**:
- Long round-robin chains (5-6 AIs contributing sequentially)
- Full STEP file pasting (give AI complete picture)
- Architecture review (needs to see everything)

**Example**:
```
[Paste entire STEP file - 8,000 lines]

Your role: Review this complete development chain for
architectural consistency.
```

**Result**: GLM can see all 8,000 lines, find inconsistencies across rounds

---

### Strategy for HUGE Context AIs (Codex)

**Advantage**: Can hold large codebases (~80,000 words)

**Use for**:
- Implementation (needs to see all related code)
- Refactoring (needs full file context)
- Integration (needs to see multiple modules)

**Example**:
```
[Paste 5 related Python files - 3,000 lines total]

Task: Implement the WorldModelBridge to connect these components.
```

**Result**: Codex sees all files, implements bridge correctly

---

### Strategy for MODERATE Context AIs (Claude, GPT-4 earlier versions)

**Advantage**: Still substantial (~25,000-50,000 words)

**Use for**:
- Single module implementation
- Focused review (one component)
- Documentation (one feature)

**Limitation**: Can't hold entire large chain

**Workaround**:
```
[Paste summary of previous work - 500 lines]
[Paste detailed section relevant to current task - 1,000 lines]

Task: [Focused task on this section]
```

**Result**: AI has enough context for focused work, doesn't need everything

---

## Context Exhaustion: Warning Signs

**Signs your AI is running out of context**:

1. **Contradicts earlier decisions**
   - "Let's use approach X" (when earlier decided on Y)

2. **Asks already-answered questions**
   - "What architecture are we using?" (already discussed)

3. **Forgets constraints**
   - Suggests CPU fallback (when sovereignty principle requires GPU-only)

4. **Repeats itself**
   - Regenerates same code with minor variations

5. **Generic responses**
   - Loses specificity, gives boilerplate answers

**What to do**:
1. **Stop the current conversation**
2. **Assemble relevant context** (Technique 1-4 above)
3. **Start NEW conversation** with assembled context
4. **Continue from checkpoint**

---

## Best Practices from K3D

### Practice 1: Plan Round Boundaries

**Before starting chain**:
- Estimate: How many AIs will contribute?
- Calculate: Total output (est. lines per AI × # of AIs)
- If > 10,000 lines: Plan for 2 rounds

**Round 1**: Foundation (design, review, enhance)
**Round 2**: Implementation (code, test, document)

**Between rounds**: Daniel creates summary (200-500 lines)

---

### Practice 2: Use King-Context AIs for Long Chains

**If your chain will be long** (5+ AIs, 5,000+ lines):
- Use GLM, Qwen Max, or Qwen Code for later stages
- They can hold the full chain in context
- No need for aggressive summarization

**If using moderate-context AIs**:
- Break into shorter chains (3-4 AIs max)
- Create summaries between chains
- Use incremental context building

---

### Practice 3: Create Summaries Proactively

**After every 3-4 AI contributions**:
- Pause
- Create summary (Daniel does this manually)
- Summary = Key decisions + rationale (not full text)

**Summary template**:
```markdown
## Summary: Rounds 1-4 (AI #1 through AI #4)

**Architecture decisions**:
1. [Decision 1] - Rationale: [Why] - Source: [AI #X]
2. [Decision 2] - Rationale: [Why] - Source: [AI #Y]

**Key components**:
- [Component 1]: [Purpose]
- [Component 2]: [Purpose]

**Constraints established**:
- [Constraint 1]
- [Constraint 2]

**Still open questions**:
- [Question 1]
- [Question 2]
```

**Use this summary**: Paste to AI #5, AI #6, etc. instead of full text from AI #1-4

---

### Practice 4: Test Context Fitness

**Before pasting large context to AI**:
- Count lines (rough estimate: 1 line ≈ 10 tokens)
- Check against AI's context window
- If > 80% of window: Trim or summarize

**Trim strategies**:
1. Remove redundant explanations (keep conclusions)
2. Remove code comments (if well-named variables)
3. Remove non-normative examples (keep requirements)
4. Paste only changed sections (not full file)

---

## Advanced: Context-Aware AI Selection

### For Long Chains: Start with Moderate, Escalate to King

**Rounds 1-2** (design, review):
- Use moderate context AIs (ChatGPT, Claude)
- Context is manageable (<5,000 lines)

**Round 3+** (integration, theory):
- Switch to king-context AIs (GLM, Qwen)
- Context is growing (>5,000 lines)
- King-context AIs can hold full chain

**Advantage**: Cost optimization (moderate AIs often cheaper)

---

### For Code Implementation: Use Huge Context

**When implementing**:
- Use Codex (huge context + VSCode integration)
- Can see entire codebase (~10,000 lines)
- No need for selective pasting

**When reviewing**:
- Use king-context AI (GLM) to review implementation
- Paste: Original design + implementation + test results
- King context holds all three

---

## Troubleshooting Context Issues

### Issue 1: "AI seems confused about earlier decisions"

**Diagnosis**: Context loss mid-conversation

**Fix**:
1. Start new conversation
2. Paste: Briefing + summary of earlier work + current state
3. Explicitly state: "Continue from here, maintaining these decisions: [list]"

---

### Issue 2: "AI keeps repeating same suggestions"

**Diagnosis**: Context window full, AI in degraded mode

**Fix**:
1. Stop current conversation (it's not recovering)
2. Create summary of current state
3. Start fresh conversation with summary
4. Continue with reduced context

---

### Issue 3: "I don't know what context to paste"

**Diagnosis**: Need to identify relevant vs. irrelevant context

**Fix**: Ask yourself:
- What decisions does AI need to know?
- What constraints must AI respect?
- What previous work will AI build on?
- What can AI safely ignore?

**Paste**: Only answers to first 3 questions

---

### Issue 4: "Even king-context AI is struggling"

**Diagnosis**: Chain is MASSIVE (>100,000 lines), even GLM/Qwen overwhelmed

**Fix**:
1. Break into modules (separate chains per module)
2. Create integration chain (synthesizes modules)
3. Use architectural summaries (not full text)

**K3D example**: K3D has phases (A-H), each phase has separate chains

---

## Context Management Checklist

**Before starting multi-vibe chain**:
- [ ] Estimated total output (lines)?
- [ ] Which AI has sufficient context window?
- [ ] Plan round boundaries (if > 5,000 lines)?
- [ ] Prepared briefing (reusable context)?

**During chain** (after each AI):
- [ ] Current total lines?
- [ ] Context still manageable?
- [ ] Time to create summary?
- [ ] Next AI has sufficient context window?

**When context issues appear**:
- [ ] Identify what AI "forgot"
- [ ] Create focused summary (not full text)
- [ ] Start new conversation with summary
- [ ] Explicitly state "Continue from here"

---

## Key Takeaways

1. **Context windows have limits** - Even king-context AIs (GLM, Qwen)
2. **Daniel's role** - "Message board software" actively manages context
3. **Techniques**:
   - Selective pasting (not everything)
   - Citing sources (AI knows provenance)
   - Message board threading (quote relevant parts)
   - Incremental building (add context as needed)
4. **Prevention**:
   - Plan round boundaries
   - Use king-context AIs for long chains
   - Create summaries proactively
5. **Recovery**:
   - Stop, summarize, restart fresh
   - Paste checkpoint + continue

**Remember**: Context management is a **required orchestration skill**, not optional. Daniel's 125+ chains prove it's learnable!

---

**Next**: [Copy-Paste Discipline](./07_copy_paste_discipline.md) - The workflow that makes it all work

**See also**: [AI Selection Guide](./05_ai_selection_guide.md) - Which AI for context window requirements
