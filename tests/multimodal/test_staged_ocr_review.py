"""Length is a review signal; legitimate content is never pruned by heuristics."""
from copy import deepcopy
from mandol_competition.multimodal.vlm.staged_ocr import primary_review_reason, recovery_crops
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.vlm.openai_compat import build_vision_client
from PIL import Image, ImageDraw
import pytest
from mandol_competition.multimodal.vlm.ocr_recovery import review_numeric_repetition, OCRNumericRepeatReview


def test_numeric_generation_loop_is_reviewed_below_a_thousand_tokens_without_altering_lines():
    lines = [' '.join(['0'] * 80)]
    original = deepcopy(lines)
    assert primary_review_reason(lines, 400) == 'numeric_repetition'
    assert lines == original
    # An ordinary invoice can repeat a total in several physically distinct rows.
    amounts = ['Total 0.00', 'Paid 0.00', 'Balance 0.00', '0 10 20 30']
    assert primary_review_reason(amounts, 80) is None


def test_long_clear_document_requests_complete_reread_and_is_not_truncated():
    lines = ['Detailed literal document row %d: keep the entire source sentence.' % n for n in range(90)]
    original = deepcopy(lines)
    assert primary_review_reason(lines, 1800) == 'long_output'
    assert lines == original and len(lines) == 90


def test_localizer_bbox_cannot_remove_headers_footers_or_side_text_from_recovery():
    wire = dict(inspected_entire_image=True, assessment='text_regions', regions=[dict(
        region_id='main-panel', bbox=[200, 200, 800, 800], readability='readable', reason='visible center')])
    tasks = recovery_crops((450, 1954), wire, 'numeric_repetition')
    cursor = 0
    for task in tasks:
        left, top, right, bottom = task.ownership_box
        assert left == 0 and right == 450 and top == cursor
        assert task.source_box[1] <= top < bottom <= task.source_box[3]
        cursor = bottom
    assert cursor == 1954 and 1 < len(tasks) <= 12


def test_multimodal_default_and_stale_runtime_override_both_keep_provider_calls_within_1024(monkeypatch):
    monkeypatch.setenv('MANDOL_GENERATION_PROVIDER', 'configured')
    monkeypatch.setenv('ARTIFICIAL_VISION_KEY', 'not-a-real-credential')
    assert MultimodalConfig().vlm_output_tokens == MultimodalConfig().vlm_max_output_tokens == 1024
    for initial, maximum in [(1024, 1024), (4096, 8192), (16384, 16384)]:
        config = MultimodalConfig(vlm_base_url='https://artificial.invalid/v1',
            vlm_api_key_env='ARTIFICIAL_VISION_KEY', vlm_output_tokens=initial, vlm_max_output_tokens=maximum)
        client = build_vision_client(config)
        try:
            assert client.output_tokens == client.max_output_tokens == 1024
        finally:
            client.close()


def test_disjoint_inputs_reconstruct_every_source_pixel_once_and_prefer_blank_cuts():
    image = Image.new('RGB', (450, 954), 'white')
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 309, 440, 327), fill='black')
    draw.rectangle((10, 625, 440, 645), fill='black')
    wire = dict(inspected_entire_image=True, assessment='text_regions', regions=[dict(
        region_id='text', bbox=[0, 0, 1000, 1000], readability='readable', reason='synthetic text rows')])
    try:
        tasks = recovery_crops(image.size, wire, 'long_output', image=image)
        parts = []
        for task in tasks:
            assert task.disjoint_input and task.source_box == task.ownership_box
            crop = image.crop(task.source_box)
            try:
                parts.append(crop.tobytes())
            finally:
                crop.close()
        assert b''.join(parts) == image.tobytes()
        assert all(not (309 <= task.source_box[3] <= 327 or 625 <= task.source_box[3] <= 645)
            for task in tasks[:-1])
    finally:
        image.close()


@pytest.mark.parametrize('lines', [
    ['19 18 15 ' + '1 9 18 15 ' * 4],
    ['1 ' + '0 ' * 15] * 4,
])
def test_varied_prefix_numeric_loops_request_independent_read_without_modifying_source(lines):
    original = deepcopy(lines)
    with pytest.raises(OCRNumericRepeatReview):
        review_numeric_repetition(lines)
    assert lines == original


def test_literal_consecutive_numbers_are_not_a_pattern_loop_verdict():
    lines = [' '.join(str(n) for n in range(1, 129))]
    original = deepcopy(lines)
    review_numeric_repetition(lines)
    assert lines == original
