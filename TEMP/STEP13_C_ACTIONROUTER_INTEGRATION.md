# Step 13-C: ActionRouter Integration with ActionBuffer

**Priority**: 2 (Second track)
**Status**: Ready to Execute (Unblocked by Step 12 ActionBuffer)
**Dependencies**: Step 12 Complete ✓ (ActionBuffer now populated)
**Estimated Effort**: 2-3 sessions

---

## Objective

Wire ThinkingTagBridge's newly populated ActionBuffer to the ActionRouter, enabling seamless multi-modal action dispatch with confidence propagation and tablet replay logging.

---

## Current State

**What's Working** (as of Step 12):
- ✓ ThinkingTagBridge populates ActionBuffer (288 bytes) in every inference
- ✓ ActionBuffer contract validated (`action_types.py`)
- ✓ ActionBuffer includes: action_type, confidence, curiosity, nav/dialogue/memory/tablet fields
- ✓ Modal signature encoded in bitfield

**What's Missing**:
- ActionRouter doesn't consume ActionBuffer yet
- Multi-modal confidence not propagated to action execution
- Tablet replay logging not wired
- SleepTime consolidation tickets incomplete

---

## Phase 1: Wire ActionBuffer to ActionRouter

### 1.1 Locate ActionRouter Implementation
**Task**: Find existing ActionRouter code

**Search for**:
- `ActionRouter` class definition
- `decode_actions.ptx` usage
- Action dispatch logic
- Current input format

**Expected locations**:
- `knowledge3d/cranium/actions/` directory
- `knowledge3d/cranium/ptx_runtime/` (if sovereign implementation)

### 1.2 Create ActionBuffer Consumer Interface
**File**: `knowledge3d/cranium/actions/action_router.py` (MODIFY or CREATE)

**Changes**:
```python
from knowledge3d.cranium.actions.action_types import ActionBuffer, ActionType

class ActionRouter:
    """Routes actions from ThinkingTagBridge to execution systems."""

    def dispatch(self, action_buffer: ActionBuffer) -> ActionResult:
        """
        Dispatch action from ThinkingTagBridge ActionBuffer.

        Args:
            action_buffer: 288-byte action buffer from inference

        Returns:
            ActionResult with success/failure status
        """
        # Dispatch based on action_type
        if action_buffer.action_type == ActionType.NAV_MOVE:
            return self._dispatch_navigation(action_buffer)
        elif action_buffer.action_type == ActionType.NAV_LOOK:
            return self._dispatch_look(action_buffer)
        elif action_buffer.action_type == ActionType.DIALOGUE:
            return self._dispatch_dialogue(action_buffer)
        elif action_buffer.action_type == ActionType.WRITE_MEM:
            return self._dispatch_memory_write(action_buffer)
        elif action_buffer.action_type == ActionType.UPDATE_TABLET:
            return self._dispatch_tablet_update(action_buffer)
        else:
            return ActionResult(
                action_type=ActionType.NO_ACTION,
                confidence=0.0,
                curiosity=0.0,
                success=False
            )
```

**Success Criteria**: ActionRouter can parse ActionBuffer and route to appropriate handler

### 1.3 Implement Action Handlers
**File**: `knowledge3d/cranium/actions/action_router.py` (MODIFY)

**Handlers to implement**:
- [ ] `_dispatch_navigation()` - Route nav_position/direction to navigation system
- [ ] `_dispatch_look()` - Update camera orientation
- [ ] `_dispatch_dialogue()` - Route dialogue tokens to text generation
- [ ] `_dispatch_memory_write()` - Write to Galaxy memory
- [ ] `_dispatch_tablet_update()` - Update tablet state

**Placeholder implementations OK** - focus on interface contracts first

---

## Phase 2: Multi-Modal Confidence Propagation

### 2.1 Add Confidence Tracking to ActionRouter
**File**: `knowledge3d/cranium/actions/action_router.py` (MODIFY)

**Add confidence tracking**:
```python
class ActionRouter:
    def __init__(self):
        self.confidence_history = []  # Track confidence over time
        self.confidence_threshold = 0.5  # Reject actions below threshold

    def dispatch(self, action_buffer: ActionBuffer) -> ActionResult:
        """Dispatch with confidence gating."""
        # Gate actions based on confidence
        if action_buffer.confidence < self.confidence_threshold:
            logger.warning(
                f"Action rejected: confidence {action_buffer.confidence:.2f} "
                f"below threshold {self.confidence_threshold:.2f}"
            )
            return ActionResult(
                action_type=action_buffer.action_type,
                confidence=action_buffer.confidence,
                curiosity=action_buffer.curiosity,
                success=False
            )

        # Record confidence
        self.confidence_history.append({
            'timestamp': time.time(),
            'action_type': action_buffer.action_type,
            'confidence': action_buffer.confidence,
            'curiosity': action_buffer.curiosity
        })

        # Dispatch action
        result = self._route_action(action_buffer)

        return result
```

**Success Criteria**: Low-confidence actions rejected, history tracked

### 2.2 Add Modal Signature Analysis
**File**: `knowledge3d/cranium/actions/action_router.py` (MODIFY)

**Track modal signature patterns**:
```python
def _analyze_modal_patterns(self, action_buffer: ActionBuffer):
    """
    Analyze which modal combinations produce high-confidence actions.

    Uses ActionBuffer.modalities bitfield to track patterns.
    """
    # Decode modal signature from bitfield
    modalities = self._decode_modalities(action_buffer.flags)

    # Track success rate per modal combination
    modal_key = tuple(sorted(modalities))
    if modal_key not in self.modal_success_rates:
        self.modal_success_rates[modal_key] = {'successes': 0, 'total': 0}

    # Update after action execution
    # (Called from dispatch result callback)
```

**Success Criteria**: Modal patterns tracked for future optimization

---

## Phase 3: Tablet Replay Logging

### 3.1 Create Tablet Logger
**File**: `knowledge3d/cranium/tablet/tablet_logger.py` (NEW or MODIFY)

**Interface**:
```python
class TabletLogger:
    """Logs actions for tablet replay and debugging."""

    def __init__(self, log_dir: str = "logs/tablet_replay"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = self._create_session_log()

    def log_action(self, action_buffer: ActionBuffer, result: ActionResult):
        """
        Log action and result for replay.

        Format: JSON lines (one action per line)
        """
        log_entry = {
            'timestamp': time.time(),
            'action_type': action_buffer.action_type.name,
            'confidence': float(action_buffer.confidence),
            'curiosity': float(action_buffer.curiosity),
            'result': {
                'success': result.success,
                'error': result.error if hasattr(result, 'error') else None
            },
            # Include action-specific fields based on type
            'data': self._extract_action_data(action_buffer)
        }

        with open(self.current_session, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
```

**Success Criteria**: All actions logged to JSON lines file

### 3.2 Wire Tablet Logger to ActionRouter
**File**: `knowledge3d/cranium/actions/action_router.py` (MODIFY)

**Integration**:
```python
class ActionRouter:
    def __init__(self, enable_tablet_logging: bool = True):
        self.tablet_logger = TabletLogger() if enable_tablet_logging else None

    def dispatch(self, action_buffer: ActionBuffer) -> ActionResult:
        result = self._route_action(action_buffer)

        # Log to tablet
        if self.tablet_logger:
            self.tablet_logger.log_action(action_buffer, result)

        return result
```

**Success Criteria**: Actions automatically logged when tablet logging enabled

---

## Phase 4: SleepTime Consolidation Integration

### 4.1 Wire ActionBuffer to SleepTime
**File**: `knowledge3d/cranium/ptx_runtime/sleep_time_compute.py` (MODIFY)

**Current state**: SleepTime has stubs for fused head integration

**Changes**:
```python
def _process_thinking_tags(self, thinking_output):
    """Process thinking tag output during sleep consolidation."""

    # STEP 13-C: Extract ActionBuffer from thinking output
    if hasattr(thinking_output, 'action_buffer') and thinking_output.action_buffer:
        action_buffer = thinking_output.action_buffer

        # Log consolidation actions
        if action_buffer.action_type == ActionType.WRITE_MEM:
            # Memory consolidation action
            self._consolidate_memory(action_buffer)
        elif action_buffer.action_type == ActionType.UPDATE_TABLET:
            # Update tablet during sleep
            self._update_tablet_sleep(action_buffer)

    # Continue with existing thinking tag processing
    # ...
```

**Success Criteria**: SleepTime can process ActionBuffer during consolidation

### 4.2 Add Curiosity-Based Consolidation
**File**: `knowledge3d/cranium/ptx_runtime/sleep_time_compute.py` (MODIFY)

**Use ActionBuffer.curiosity field**:
```python
def _prioritize_consolidation(self, action_buffers: List[ActionBuffer]):
    """
    Prioritize memory consolidation based on curiosity scores.

    High-curiosity memories consolidated first (novel/surprising).
    """
    # Sort by curiosity (descending)
    sorted_actions = sorted(
        action_buffers,
        key=lambda ab: ab.curiosity,
        reverse=True
    )

    # Consolidate top 20% highest curiosity
    consolidation_threshold = 0.7
    high_curiosity = [
        ab for ab in sorted_actions
        if ab.curiosity > consolidation_threshold
    ]

    return high_curiosity
```

**Success Criteria**: Curiosity score influences consolidation priority

---

## Phase 5: Testing & Validation

### 5.1 Create ActionRouter Integration Tests
**File**: `tests/test_step13_actionrouter_integration.py` (NEW)

**Coverage**:
- [ ] ActionBuffer dispatch to all 5 action types
- [ ] Confidence gating (reject low-confidence)
- [ ] Tablet logging (verify JSON output)
- [ ] Modal pattern tracking
- [ ] Error handling (invalid action types)

**Success Criteria**: 15+ tests passing

### 5.2 Create End-to-End Flow Test
**File**: `tests/test_step13_inference_to_action.py` (NEW)

**Flow**:
```python
def test_inference_to_action_flow():
    """Test full flow: ThinkingTag → ActionBuffer → ActionRouter → Execution"""

    # 1. Run inference
    bridge = ThinkingTagBridge()
    result = bridge.inference(input_embedding, modal_signature=['text'])

    # 2. Verify ActionBuffer populated
    assert result.action_buffer is not None
    assert result.action_buffer.confidence > 0.0

    # 3. Dispatch to ActionRouter
    router = ActionRouter()
    action_result = router.dispatch(result.action_buffer)

    # 4. Verify execution
    assert action_result.success is True or action_result.success is False  # Either is valid

    # 5. Verify tablet logging
    assert router.tablet_logger.current_session.exists()
```

**Success Criteria**: End-to-end flow works without errors

### 5.3 Performance Validation
**File**: `tests/benchmarks/test_actionrouter_latency.py` (NEW)

**Benchmarks**:
- [ ] ActionBuffer dispatch latency (target: <5µs)
- [ ] Confidence gating overhead (target: <1µs)
- [ ] Tablet logging latency (target: <100µs, non-blocking)
- [ ] Total overhead (target: <10µs added to inference)

**Success Criteria**: <10µs total overhead maintains <35µs inference budget

---

## Integration Points

### With ThinkingTagBridge
- **Input**: `ThinkingTagOutput.action_buffer` (288 bytes)
- **Contract**: ActionBuffer always populated (or None if unavailable)
- **Frequency**: Every inference call

### With ActionRouter
- **Input**: ActionBuffer from ThinkingTag
- **Output**: ActionResult (success/failure + metadata)
- **Error Handling**: Graceful degradation if dispatch fails

### With TabletLogger
- **Input**: ActionBuffer + ActionResult
- **Output**: JSON lines log file
- **Storage**: `logs/tablet_replay/session_YYYYMMDD_HHMMSS.jsonl`

### With SleepTime
- **Input**: ActionBuffer during consolidation
- **Usage**: Memory prioritization via curiosity score
- **Integration**: Optional (SleepTime can work without ActionBuffer)

---

## Success Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| ActionRouter consumes ActionBuffer | ✓ | Pending |
| All 5 action types dispatched | ✓ | Pending |
| Confidence gating implemented | threshold=0.5 | Pending |
| Tablet logging working | JSON lines format | Pending |
| Modal pattern tracking active | ✓ | Pending |
| SleepTime integration | Optional curiosity-based | Pending |
| Tests passing | 15+ integration tests | Pending |
| Performance overhead | <10µs | Pending |

---

## File Structure

```
knowledge3d/cranium/actions/
├── action_types.py                    # Existing (ActionBuffer, ActionType)
├── action_router.py                   # MODIFY (add ActionBuffer dispatch)
└── tablet_logger.py                   # NEW (tablet replay logging)

knowledge3d/cranium/tablet/
└── tablet_logger.py                   # NEW (alternative location)

knowledge3d/cranium/ptx_runtime/
└── sleep_time_compute.py              # MODIFY (ActionBuffer integration)

tests/
├── test_step13_actionrouter_integration.py   # NEW (unit tests)
├── test_step13_inference_to_action.py        # NEW (end-to-end)
└── benchmarks/
    └── test_actionrouter_latency.py          # NEW (performance)

logs/tablet_replay/                    # NEW (tablet log directory)
└── session_YYYYMMDD_HHMMSS.jsonl     # Generated at runtime
```

---

## Execution Plan

### Session 1: Core Integration
1. Locate existing ActionRouter code
2. Add ActionBuffer dispatch method
3. Implement 5 action handlers (stubs OK)
4. Create integration tests
5. Verify basic dispatch works

### Session 2: Confidence & Logging
1. Add confidence gating logic
2. Implement modal pattern tracking
3. Create TabletLogger class
4. Wire tablet logging to ActionRouter
5. Test confidence thresholds

### Session 3: SleepTime & Validation
1. Integrate ActionBuffer with SleepTime
2. Add curiosity-based consolidation
3. Create end-to-end flow test
4. Run performance benchmarks
5. Document integration

---

## Deliverables

### Code
- [ ] ActionRouter ActionBuffer dispatch (~150 lines)
- [ ] TabletLogger implementation (~100 lines)
- [ ] SleepTime integration (~50 lines)
- [ ] 3 test files (~300 lines total)

### Documentation
- [ ] ActionRouter API documentation
- [ ] Tablet logging format specification
- [ ] Integration diagram (ThinkingTag → ActionRouter → Systems)
- [ ] Performance impact report

### Logs
- [ ] Example tablet replay log
- [ ] Confidence tracking statistics
- [ ] Modal pattern success rates

---

## Notes

- ActionBuffer is already populated by ThinkingTagBridge (Step 12 ✓)
- Focus on interface contracts before implementation details
- Tablet logging should be non-blocking (async if needed)
- Confidence threshold should be configurable
- Modal pattern tracking is for future optimization (not critical path)

---

**Ready to Execute**: Yes ✓ (Unblocked by Step 12)
**Next Step**: Locate ActionRouter code and add ActionBuffer dispatch
**Estimated Completion**: 2-3 sessions
