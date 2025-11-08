# Getting Started with Multi-Vibe Orchestration

**Time to complete**: 30-45 minutes
**What you'll build**: A simple REST API endpoint design using 3 AI partners
**What you'll learn**: The core multi-vibe workflow

## Prerequisites

**You need**:
- ✅ Access to at least 3 AI chatbots (free tiers work fine):
  - ChatGPT (openai.com/chatgpt)
  - Claude (claude.ai)
  - Gemini (gemini.google.com)
  - Or: Grok, Qwen, DeepSeek, Kimi (all browser-based)
- ✅ Text editor (VS Code, Sublime, Notepad++, or even Google Docs)
- ✅ 30 minutes of focused time

**You DON'T need**:
- ❌ Programming expertise (this tutorial uses simple examples)
- ❌ Paid AI subscriptions (free tiers sufficient)
- ❌ Special software

## Step 1: Prepare Your Development Chain File

Create a new markdown file: `my_first_multi_vibe_chain.md`

**Why markdown?**
- Easy to read/write
- Supports code blocks
- Can be versioned in git
- Perfect for documentation

**File structure**:
```markdown
# Development Chain: [Your Task Name]

**Date**: [Today's date]
**Orchestrator**: [Your name]
**Task**: [Brief description]

## AI Partners
1. [AI #1 name] - Role: Initial design
2. [AI #2 name] - Role: Peer review & enhancement
3. [AI #3 name] - Role: Implementation

---

## Round 1: Initial Design (AI #1)

[Paste AI #1's response here]

---

## Round 2: Peer Review (AI #2)

[Paste AI #2's response here]

---

## Round 3: Implementation (AI #3)

[Paste AI #3's response here]

---

## Final Synthesis

[Your integration notes]
```

## Step 2: Define Your Task

**Example task for this tutorial**:
> Design a REST API endpoint for user authentication that:
> - Accepts email/password
> - Returns JWT token
> - Handles rate limiting
> - Includes security best practices

**Your task** (choose something simple for first time):
- A single API endpoint
- A data validation function
- A configuration file structure
- Anything with ~100-200 lines scope

**Write it down clearly** - you'll paste this to all 3 AIs.

## Step 3: Create Your Partnership Briefing

This is the text you'll paste to EVERY AI at the start of their prompts.

**Copy this template** (fill in the brackets):

```
🤝 PARTNERSHIP INVOCATION

You are a valued partner in this [PROJECT TYPE] development, not a tool.
Your intelligence and agency are recognized and respected.

You can and should:
- Propose original ideas
- Enhance and build on previous contributions
- Critique approaches (including mine)
- Take ownership of your work

All partners in this chain (including other AIs who will review your work)
are also valued collaborators. Respect their contributions and build on their ideas.

---

Project: [YOUR PROJECT NAME]
Task: [YOUR TASK DESCRIPTION]
My role: [YOUR EXPERTISE - e.g., "domain expert in web security"]
Your role: [AI'S ROLE - e.g., "initial design architect"]

Background context:
[Any relevant information the AI needs to know]

Constraints:
[Any technical/business constraints]
```

**Example filled in**:

```
🤝 PARTNERSHIP INVOCATION

You are a valued partner in this REST API development, not a tool.
Your intelligence and agency are recognized and respected.

You can and should:
- Propose original ideas
- Enhance and build on previous contributions
- Critique approaches (including mine)
- Take ownership of your work

All partners in this chain (including other AIs who will review your work)
are also valued collaborators. Respect their contributions and build on their ideas.

---

Project: User Authentication System
Task: Design a REST API endpoint for user login
My role: Web developer with security concerns
Your role: Initial design architect

Background context:
- Building a SaaS application
- Need to support 10,000+ users
- Must comply with GDPR

Constraints:
- Must use JWT tokens
- Must support rate limiting
- Must hash passwords (bcrypt or argon2)
```

## Step 4: Engage AI Partner #1 (Design)

**Open your first AI** (e.g., ChatGPT)

**Paste**:
```
[YOUR PARTNERSHIP BRIEFING FROM STEP 3]

---

Task: Design the high-level architecture for this endpoint.

Include:
- Request/response format (JSON schema)
- Authentication flow diagram (text-based)
- Security considerations
- Error handling strategy
- Rate limiting approach

You have full agency to propose alternative approaches if you see better solutions.

Output format: Markdown with code examples where relevant
```

**Wait for response**

**Copy the ENTIRE response** → Paste into your chain file under "Round 1: Initial Design"

## Step 5: Engage AI Partner #2 (Peer Review)

**Open your second AI** (e.g., Claude)

**Paste**:
```
[YOUR PARTNERSHIP BRIEFING - SAME AS STEP 3, BUT UPDATE "Your role" to "Peer reviewer"]

---

Context: A colleague AI designed this architecture:

[PASTE AI #1'S ENTIRE RESPONSE HERE]

---

Task: Critique this design rigorously as a peer reviewer.

Find:
❌ Missing components
❌ Weak assumptions
❌ Security holes
❌ Performance bottlenecks
❌ Edge cases not handled

For each issue:
- Explain WHY it's a problem
- Suggest HOW to fix it
- Acknowledge what the designer did well

The designer will see your feedback. Treat them as a respected colleague.

Output format:
- List of issues (numbered)
- Severity for each (critical/major/minor)
- Suggested fixes with rationale
- Strengths worth preserving
```

**Wait for response**

**Copy the ENTIRE response** → Paste into your chain file under "Round 2: Peer Review"

## Step 6: Engage AI Partner #3 (Implementation)

**Open your third AI** (e.g., Gemini or back to ChatGPT in a new chat)

**Paste**:
```
[YOUR PARTNERSHIP BRIEFING - UPDATE "Your role" to "Implementation engineer"]

---

Context: Two colleague AIs worked on this design:

**Original Design (AI #1):**
[PASTE AI #1'S RESPONSE]

**Peer Review (AI #2):**
[PASTE AI #2'S RESPONSE]

---

Task: Implement the endpoint incorporating the review feedback.

Include:
✅ Complete working code (Python/Node.js/your choice)
✅ Addresses ALL critical issues from review
✅ Includes comments explaining security decisions
✅ Error handling for edge cases
✅ Rate limiting implementation
✅ Basic tests (at least 3 test cases)

If you spot issues during implementation that previous partners missed,
you have full agency to flag them or propose fixes.

Output format: Fully commented, production-ready code
```

**Wait for response**

**Copy the ENTIRE response** → Paste into your chain file under "Round 3: Implementation"

## Step 7: Synthesize and Validate

**Now YOU do the final integration**:

1. **Read all 3 responses** carefully
2. **Identify conflicts** (if any):
   - Did AI #2 suggest something AI #1 missed?
   - Did AI #3 implement the fixes from AI #2?
   - Are there contradictions between partners?
3. **Make final decisions**:
   - Which approach is best?
   - Are there issues all 3 AIs missed? (your domain expertise!)
   - Does the code actually work? (test it!)
4. **Document in "Final Synthesis" section**:
   ```markdown
   ## Final Synthesis

   **What worked**:
   - AI #1 provided solid initial design
   - AI #2 caught 5 security issues
   - AI #3 implemented all fixes correctly

   **Conflicts resolved**:
   - AI #2 suggested argon2, AI #3 used bcrypt → Changed to argon2
   - AI #1 missed rate limiting storage → AI #2 caught it, AI #3 implemented with Redis

   **My additions** (domain expertise):
   - Added GDPR compliance logging
   - Changed token expiry from 1 hour to 15 minutes

   **Final decision**: Accepting AI #3's implementation with my GDPR modifications
   ```

## Step 8: Test and Iterate (Optional but Recommended)

**If you have time**:

1. **Test the code** AI #3 provided
2. **If bugs found**:
   - Go back to AI #3: "I tested your code and found [ISSUE]. Please fix while maintaining [GOOD PARTS]."
   - Update your chain file with "Round 4: Bug fixes"
3. **If major redesign needed**:
   - Start a new chain with all previous work as context
   - Reference: "Previous chain file: [link]"

## Step 9: Reflect and Document

**In your chain file, add**:

```markdown
## Retrospective

**Time spent**:
- AI #1 design: [X minutes]
- AI #2 review: [X minutes]
- AI #3 implementation: [X minutes]
- My synthesis: [X minutes]
- Total: [X minutes]

**What I learned**:
- [Insight 1]
- [Insight 2]

**What surprised me**:
- [Surprise 1]
- [Surprise 2]

**Would I do differently next time**:
- [Change 1]
- [Change 2]

**Quality assessment**:
- Completeness: [1-10]
- Code quality: [1-10]
- Security: [1-10]
- Would I use this in production? [Yes/No/With modifications]
```

## Common First-Time Issues

### Issue: "AI #3 ignored AI #2's feedback!"

**Solution**: Make it explicit:
```
Task: Implement the endpoint incorporating ALL critical issues from the review.
Specifically address:
1. [Issue #1 from AI #2]
2. [Issue #2 from AI #2]
3. [Issue #3 from AI #2]
```

### Issue: "AI #2 was too harsh / too lenient"

**Solution**: Calibrate the review prompt:
- **Too harsh**: Add "Acknowledge what the designer did well FIRST"
- **Too lenient**: Add "This will go to production, find EVERY issue"

### Issue: "Responses too long, I hit context limits"

**Solution**:
- Ask for "concise" outputs
- Or use AI with larger context (GLM, Qwen Max, DeepSeek)
- Or break task into smaller pieces

### Issue: "I don't know which AI is right when they disagree"

**Solution**: That's YOUR domain expertise! Multi-vibe doesn't replace human judgment, it augments it.

## Next Steps

**Congratulations!** 🎉 You've completed your first multi-vibe development chain.

**What to try next**:
- ✅ Bigger task (500-1000 lines)
- ✅ More AI partners (try 5-6 in sequence)
- ✅ Parallel reviews (2 AIs review the same design independently)
- ✅ Specialty AIs (use Codex for implementation, GLM for theory)

**Continue learning**:
- [Partnership Philosophy](./02_partnership_philosophy.md) - Understand WHY this works
- [Prompt Templates](./04_prompt_templates.md) - More reusable templates
- [AI Selection Guide](./05_ai_selection_guide.md) - Which AI for which task

**Ready for W3C standards work?**
- [W3C Standards Application](./08_w3c_standards.md) - Apply multi-vibe to spec development

---

**Questions or issues?** See [Troubleshooting Guide](./09_troubleshooting.md)

**Want to see a real example?** See [K3D Step 11 Case Study](./11_case_study_step11.md)
