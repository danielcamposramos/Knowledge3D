# Phase 2: Bidirectional Reasoning Architecture

**User Insight** (January 13, 2026):
> "Humans... also verbally say those terms... and often read it backwards, 
> meaning we start constructing the problem information from the last info 
> to the first one."

## Three Cognitive Modes

### 1. Forward Pass (Phase 1 - IMPLEMENTED)
**Direction**: Problem → Solution
**Process**: Parse text → AST → Execute rules → Numeric result
**Example**: "(3x-4)/(2x+3) at x=1" → 0.68

### 2. Backward Explanation (Phase 2 - PLANNED)
**Direction**: Solution → Reasoning
**Process**: Result → Trace steps → Verbal explanation
**Example**: "0.68 because I applied quotient rule to f=3x-4, g=2x+3..."
**Purpose**: Explainability, shadow copy learning, user trust

### 3. Inverse Reading (Phase 2 - PLANNED)
**Direction**: Goal → Problem Construction
**Process**: Identify goal ("rate of change") → Work backwards through givens
**Example**: "Find rate of volume change" → "Volume = πr²h" → "Need dV/dt"
**Purpose**: Complex word problem solving, goal-directed reasoning

## Architecture Requirements (Phase 2)

### Backward Trace Generation
- Store rule application sequence (already in forward trace)
- Convert to natural language explanation
- Link to Grammar Galaxy metadata (rule descriptions)

### Inverse Problem Construction
- Parse goal from question ("Find the rate...")
- Identify required components working backwards
- Match givens to components
- Construct forward problem from inverse analysis

### Shadow Copy Integration
- Record: "Success because quotient rule → decompose → power rules"
- Learn: "Quotient structure pattern → this navigation sequence"
- Not just "I got correct answer" but "WHY this approach worked"

## Success Criteria (Phase 2)

**Backward Explanation**:
- [ ] Generate verbal trace from execution steps
- [ ] Readable by humans (natural language)
- [ ] Matches Grammar Galaxy metadata (rule names/descriptions)

**Inverse Reading**:
- [ ] Parse goal from word problem ending
- [ ] Work backwards to identify required components
- [ ] Construct problem representation goal-first
- [ ] Accuracy improves on complex word problems

**Shadow Copy Learning**:
- [ ] Record WHY solutions worked (not just that they worked)
- [ ] TRM learns navigation patterns from explained successes
- [ ] Continual improvement from bidirectional reasoning
