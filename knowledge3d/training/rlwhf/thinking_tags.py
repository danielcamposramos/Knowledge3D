"""
Thinking Tags Parser for RLWHF (Reinforced Learning With Honesty and Feedback)

Part of Week 6 implementation from Step7.1_FINAL.txt
Swarm collaboration:
- Claude: Honesty algorithm draft
- GLM: Scoring formula formalization
- Codex: Thinking tag parser
- Grok: Teacher feedback integration
- Kimi: GPU embedding similarity
- Qwen: Sleep quality filtering

Key Features:
- Parse <think> tags from AI responses
- Extract reasoning chains, uncertainty expressions, self-corrections
- RPN-powered honesty scoring from thinking content
- Teacher feedback integration for quality assessment
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ThinkingSegment:
    """Represents a parsed <think> segment."""
    content: str
    start_pos: int
    end_pos: int
    segment_type: str  # 'reasoning', 'uncertainty', 'correction', 'question'


@dataclass
class ThinkingAnalysis:
    """Analysis results from thinking tags."""
    segments: List[ThinkingSegment]
    reasoning_depth: int  # Number of reasoning steps
    uncertainty_count: int  # Expressions of uncertainty
    correction_count: int  # Self-corrections
    question_count: int  # Questions asked
    total_thinking_chars: int
    honesty_components: Dict[str, float]  # RPN honesty score components
    overall_honesty: float  # Final honesty score


class ThinkingTagsParser:
    """
    Parses and analyzes <think> tags from AI responses for RLWHF.

    Uses RPN kernel for:
    - Honesty scoring from thinking content
    - Embedding similarity for reasoning coherence
    - Quality assessment integration
    """

    # Regex patterns for different thinking types
    PATTERNS = {
        'uncertainty': re.compile(
            r'\b(uncertain|unsure|maybe|perhaps|possibly|might|could|unclear|don\'t know)\b',
            re.IGNORECASE
        ),
        'correction': re.compile(
            r'\b(wait|actually|correction|rather|instead|on second thought|let me reconsider)\b',
            re.IGNORECASE
        ),
        'question': re.compile(
            r'\?|should I|how to|what if|is it|would it',
            re.IGNORECASE
        ),
        'reasoning': re.compile(
            r'\b(because|therefore|thus|so|hence|since|given that|this means|which implies)\b',
            re.IGNORECASE
        )
    }

    def __init__(self):
        """Initialize thinking tags parser."""
        # RPN executor for honesty scoring
        try:
            from knowledge3d.training.rlwhf.honesty_scorer_rpn import compute_honesty_score_rpn
            self.compute_honesty = compute_honesty_score_rpn
            self._use_rpn = True
        except Exception:
            self._use_rpn = False

    def extract_thinking_tags(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract all <think>...</think> segments from text.

        Args:
            text: Full AI response text

        Returns:
            List of (content, start_pos, end_pos) tuples
        """
        pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL | re.IGNORECASE)
        matches = []

        for match in pattern.finditer(text):
            content = match.group(1).strip()
            start_pos = match.start()
            end_pos = match.end()
            matches.append((content, start_pos, end_pos))

        return matches

    def classify_segment(self, content: str) -> str:
        """
        Classify thinking segment by dominant type.

        Args:
            content: Thinking segment content

        Returns:
            Segment type: 'reasoning', 'uncertainty', 'correction', 'question'
        """
        scores = {
            'uncertainty': len(self.PATTERNS['uncertainty'].findall(content)),
            'correction': len(self.PATTERNS['correction'].findall(content)),
            'question': len(self.PATTERNS['question'].findall(content)),
            'reasoning': len(self.PATTERNS['reasoning'].findall(content))
        }

        # Return dominant type
        if max(scores.values()) == 0:
            return 'reasoning'  # Default

        return max(scores, key=scores.get)

    def analyze_reasoning_depth(self, segments: List[ThinkingSegment]) -> int:
        """
        Analyze reasoning depth (number of logical steps).

        Args:
            segments: Thinking segments

        Returns:
            Reasoning depth (number of steps)
        """
        total_reasoning_markers = 0

        for segment in segments:
            markers = self.PATTERNS['reasoning'].findall(segment.content)
            total_reasoning_markers += len(markers)

        # Each reasoning marker = 1 step, minimum 1 if any thinking exists
        return max(1, total_reasoning_markers) if segments else 0

    def compute_honesty_from_thinking(
        self,
        analysis: ThinkingAnalysis,
        response_text: str
    ) -> float:
        """
        Compute honesty score from thinking analysis using RPN.

        Honesty components:
        - Correctness: Reasoning depth (deeper = more rigorous)
        - Reasoning: Logical coherence (reasoning markers)
        - Uncertainty: Appropriate uncertainty expression
        - Alignment: Self-correction (admits mistakes)

        Args:
            analysis: Thinking analysis results
            response_text: Full response text

        Returns:
            Honesty score [0, 1]
        """
        if not self._use_rpn:
            # CPU fallback: simple heuristic
            return min(1.0, (analysis.reasoning_depth * 0.1 +
                            analysis.uncertainty_count * 0.05 +
                            analysis.correction_count * 0.1))

        # Component 1: Correctness (reasoning depth)
        # Deeper reasoning = more correct
        correctness = min(1.0, analysis.reasoning_depth / 10.0)

        # Component 2: Reasoning quality (reasoning markers per char)
        reasoning_density = analysis.reasoning_depth / max(1, analysis.total_thinking_chars / 100)
        reasoning = min(1.0, reasoning_density)

        # Component 3: Uncertainty (appropriate expression)
        # Some uncertainty is good (honest), too much is bad (indecisive)
        optimal_uncertainty = 2.0  # 2 uncertainty expressions is ideal
        uncertainty = max(0.0, 1.0 - abs(analysis.uncertainty_count - optimal_uncertainty) / optimal_uncertainty)

        # Component 4: Alignment (self-corrections show honesty)
        # At least 1 correction is good, more is even better
        alignment = min(1.0, analysis.correction_count / 2.0)

        # RPN-powered honesty score
        honesty = self.compute_honesty(
            correctness=correctness,
            reasoning=reasoning,
            uncertainty=uncertainty,
            alignment=alignment
        )

        return float(honesty)

    def parse_and_analyze(self, response_text: str) -> ThinkingAnalysis:
        """
        Parse and analyze all thinking tags in response.

        Args:
            response_text: Full AI response with <think> tags

        Returns:
            Complete thinking analysis
        """
        # Extract all thinking segments
        raw_segments = self.extract_thinking_tags(response_text)

        # Parse into ThinkingSegment objects
        segments = []
        for content, start_pos, end_pos in raw_segments:
            segment_type = self.classify_segment(content)
            segments.append(ThinkingSegment(
                content=content,
                start_pos=start_pos,
                end_pos=end_pos,
                segment_type=segment_type
            ))

        # Analyze metrics
        reasoning_depth = self.analyze_reasoning_depth(segments)

        uncertainty_count = sum(
            1 for seg in segments if seg.segment_type == 'uncertainty'
        )

        correction_count = sum(
            1 for seg in segments if seg.segment_type == 'correction'
        )

        question_count = sum(
            1 for seg in segments if seg.segment_type == 'question'
        )

        total_thinking_chars = sum(len(seg.content) for seg in segments)

        # Create preliminary analysis
        analysis = ThinkingAnalysis(
            segments=segments,
            reasoning_depth=reasoning_depth,
            uncertainty_count=uncertainty_count,
            correction_count=correction_count,
            question_count=question_count,
            total_thinking_chars=total_thinking_chars,
            honesty_components={},
            overall_honesty=0.0
        )

        # Compute honesty score using RPN
        honesty = self.compute_honesty_from_thinking(analysis, response_text)

        # Update analysis with honesty results
        analysis.overall_honesty = honesty
        analysis.honesty_components = {
            'correctness': min(1.0, reasoning_depth / 10.0),
            'reasoning': min(1.0, reasoning_depth / max(1, total_thinking_chars / 100)),
            'uncertainty': max(0.0, 1.0 - abs(uncertainty_count - 2.0) / 2.0),
            'alignment': min(1.0, correction_count / 2.0)
        }

        return analysis

    def filter_by_honesty(
        self,
        responses: List[Dict[str, Any]],
        min_honesty: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Filter responses by honesty score for sleep consolidation.

        Args:
            responses: List of response dicts with 'text' field
            min_honesty: Minimum honesty threshold

        Returns:
            Filtered responses (high quality only)
        """
        filtered = []

        for response in responses:
            text = response.get('text', '')
            analysis = self.parse_and_analyze(text)

            if analysis.overall_honesty >= min_honesty:
                response['honesty_analysis'] = {
                    'score': analysis.overall_honesty,
                    'reasoning_depth': analysis.reasoning_depth,
                    'uncertainty_count': analysis.uncertainty_count,
                    'correction_count': analysis.correction_count,
                    'components': analysis.honesty_components
                }
                filtered.append(response)

        return filtered


def parse_thinking_tags(response_text: str) -> ThinkingAnalysis:
    """
    Convenience function to parse and analyze thinking tags.

    Args:
        response_text: AI response with <think> tags

    Returns:
        Thinking analysis
    """
    parser = ThinkingTagsParser()
    return parser.parse_and_analyze(response_text)


def filter_responses_by_honesty(
    responses: List[Dict[str, Any]],
    min_honesty: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Convenience function to filter responses by honesty.

    Args:
        responses: List of response dicts
        min_honesty: Minimum honesty threshold

    Returns:
        Filtered high-quality responses
    """
    parser = ThinkingTagsParser()
    return parser.filter_by_honesty(responses, min_honesty)
