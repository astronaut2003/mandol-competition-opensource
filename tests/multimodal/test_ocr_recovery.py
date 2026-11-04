"""Pure OCR recovery geometry/protocol tests; no provider, DB or image writes."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from mandol_competition.multimodal.vlm.ocr_recovery import (
    CropTask, OCRNumericRepeatReview, OCRRecoveryNeedsReview, ValidatedCrop, crop_messages, crop_response_format,
    full_image_confirmation_task, localization_messages, localization_response_format, merge_crop_results, plan_crops,
    split_crop_task, stream_review_callback, validate_crop_result,
)


def localized(*regions, assessment="text_regions", complete=True):
    return dict(inspected_entire_image=complete, assessment=assessment,
                regions=[dict(region_id="region-" + str(i), bbox=box,
                              readability="readable", reason="physical page") for i, box in enumerate(regions)])


def task(size=(1000, 1000)):
    return plan_crops(size, localized([0, 0, 1000, 1000]), margin=0)[0]


def wire(lines=(), *, assessment=None, unreadable=(), complete=True):
    return dict(inspected_entire_crop=complete,
                assessment=assessment or ("readable_text" if lines else "no_text"),
                lines=[dict(bbox=box, text=text) for box, text in lines],
                unreadable_regions=list(unreadable))


def test_unreadable_localization_is_still_close_read_and_no_text_is_explicit():
    value = localized([100, 100, 500, 900])
    value["regions"][0]["readability"] = "unreadable"
    assert len(plan_crops((2000, 2000), value)) == 2
    assert plan_crops((1000, 1000), localized(assessment="no_text")) == ()
    for failed in (localized(assessment="uncertain"), localized(complete=False),
                   localized([0, 0, 100, 100], assessment="no_text")):
        with pytest.raises(OCRRecoveryNeedsReview):
            plan_crops((1000, 1000), failed)
    with pytest.raises(OCRRecoveryNeedsReview):
        plan_crops((1000, 1000), localized(assessment="no_text"), finish_reason="length")


def test_all_regions_retained_full_width_bands_own_exact_nonoverlap_coverage():
    tasks = plan_crops((1800, 3000), localized([0, 0, 400, 1000], [600, 0, 1000, 1000]),
                       max_crop_height=1000, overlap=100, margin=0)
    assert {r for t in tasks for r in t.region_ids} == {"region-0", "region-1"}
    for region in ("region-0", "region-1"):
        bands = [t for t in tasks if region in t.region_ids]
        assert bands[0].ownership_box[1] == 0 and bands[-1].ownership_box[3] == 3000
        assert all(a.ownership_box[3] == b.ownership_box[1] for a, b in zip(bands, bands[1:]))
        assert all(t.source_box[0] == t.region_box[0] and t.source_box[2] == t.region_box[2] for t in bands)


def test_duplicate_bbox_and_excess_capacity_never_silently_drop_region():
    with pytest.raises(OCRRecoveryNeedsReview):
        plan_crops((1000, 1000), localized([1, 1, 900, 900], [1, 1, 900, 900]))
    with pytest.raises(OCRRecoveryNeedsReview):
        plan_crops((1000, 10000), localized([0, 0, 1000, 1000]), max_crop_height=100,
                   overlap=10, max_crops=12)
    with pytest.raises(OCRRecoveryNeedsReview):
        plan_crops((1000, 1000), localized([200, 0, 100, 100]))
    with pytest.raises(ValidationError):
        plan_crops((1000, 1000), localized(*([[i, 0, i + 10, 10] for i in range(9)])))


def test_overlapping_object_inventory_merged_not_recopied_or_dropped():
    tasks = plan_crops((1000, 1000), localized([100, 100, 600, 600], [500, 400, 900, 900]), margin=0)
    assert len(tasks) == 1 and set(tasks[0].region_ids) == {"region-0", "region-1"}
    assert tasks[0].source_box == (100, 100, 900, 900)


def test_dense_true_repeated_numeric_rows_are_preserved():
    t = task()
    rows = [([10, 5 + i * 5, 990, 9 + i * 5], "0 0 0 0 0 0 0 0 0 0") for i in range(128)]
    # Repetition is a review signal, never a claim that true rows are false.
    with pytest.raises(OCRNumericRepeatReview):
        validate_crop_result(t, wire(rows))
    assert rows == [([10, 5 + i * 5, 990, 9 + i * 5], "0 0 0 0 0 0 0 0 0 0") for i in range(128)]
    # Dense genuinely varying rows are not rejected merely for many numbers.
    rows = [([10, 5 + i * 5, 990, 9 + i * 5], f"{i} 0 0 0 0 0 0 0 0 0") for i in range(128)]
    value = validate_crop_result(t, wire(rows))
    result = merge_crop_results([t], [value])
    assert len(result) == 128 and result == [text for _, text in rows]
    stream_review_callback(t)(json.dumps(wire(rows)))


def test_B_style_shifted_nine_identical_numeric_rows_trigger_specific_review():
    t = task()
    rows = [([0, 30 + i * 90, 1000, 80 + i * 90], "1  19  18  15") for i in range(9)]
    payload = wire(rows)
    with pytest.raises(OCRNumericRepeatReview):
        validate_crop_result(t, payload)
    with pytest.raises(OCRNumericRepeatReview):
        stream_review_callback(t)(json.dumps(payload))
    # The complete third line suffices even before the array/reply closes.
    prefix = '{"lines":[' + ','.join(json.dumps(line) for line in payload['lines'][:3])
    with pytest.raises(OCRNumericRepeatReview):
        stream_review_callback(t)(prefix)


def test_two_numeric_rows_labels_and_currency_rows_are_not_this_review():
    t = task()
    for text in ("1 19 18 15", "SAME", "Step 3", "$19.99  $19.99  $19.99"):
        rows = [([10, 30 + i * 90, 900, 80 + i * 90], text) for i in range(2)]
        assert merge_crop_results([t], [validate_crop_result(t, wire(rows))]) == [text, text]
    for text in ("SAME", "Step 3", "$19.99  $19.99  $19.99"):
        rows = [([10, 30 + i * 90, 900, 80 + i * 90], text) for i in range(3)]
        assert len(validate_crop_result(t, wire(rows)).lines) == 3


def test_three_true_repeated_numeric_rows_can_be_reread_in_pieces_without_dedup():
    parent = task()
    literal = "1 19 18 15"
    original = wire([([10, 100, 900, 200], literal), ([10, 300, 900, 400], literal),
                     ([10, 700, 900, 800], literal)])
    before = deepcopy(original)
    with pytest.raises(OCRNumericRepeatReview):
        validate_crop_result(parent, original)
    assert original == before
    a, b = split_crop_task(parent)
    va = validate_crop_result(a, wire([([10, 180, 900, 260], literal), ([10, 530, 900, 620], literal)]))
    vb = validate_crop_result(b, wire([([10, 480, 900, 560], literal)]))
    assert merge_crop_results([a, b], [va, vb]) == [literal, literal, literal]


def test_long_same_value_numeric_row_is_reviewed_but_varying_values_not_filtered():
    t = task()
    zeros = ' '.join(['0'] * 48)
    with pytest.raises(OCRNumericRepeatReview):
        validate_crop_result(t, wire([([10, 10, 990, 50], zeros)]))
    diverse = ' '.join(str(i) for i in range(48))
    assert validate_crop_result(t, wire([([10, 10, 990, 50], diverse)])).lines[0].text == diverse


def test_repeated_physical_bbox_is_rejected_not_deduplicated_as_success():
    t = task()
    repeated = wire([([10, 10, 900, 50], "1 0 0"), ([10, 10, 900, 50], "1 0 0")])
    with pytest.raises(OCRRecoveryNeedsReview):
        validate_crop_result(t, repeated)
    with pytest.raises(OCRRecoveryNeedsReview):
        stream_review_callback(t)(json.dumps(repeated))


def test_overlap_same_physical_row_dedupes_only_if_text_identical():
    a = CropTask("a", (1000, 1000), (0, 0, 1000, 800), (0, 0, 1000, 800),
                 (0, 0, 1000, 1000), ("page",))
    b = CropTask("b", (1000, 1000), (0, 200, 1000, 1000), (0, 200, 1000, 1000),
                 (0, 0, 1000, 1000), ("page",))
    va = validate_crop_result(a, wire([([10, 500, 900, 550], "literal row")]))
    vb = validate_crop_result(b, wire([([10, 250, 900, 300], "literal row")]))
    assert merge_crop_results([a, b], [va, vb]) == ["literal row"]
    wrong = validate_crop_result(b, wire([([10, 250, 900, 300], "invented row")]))
    with pytest.raises(OCRRecoveryNeedsReview):
        merge_crop_results([a, b], [va, wrong])


def test_completed_readability_not_errors_can_make_empty_ocr():
    t = task()
    with pytest.raises(OCRRecoveryNeedsReview):
        validate_crop_result(t, wire())
    confirmation = full_image_confirmation_task((1000, 1000))
    assert confirmation.expect_text is False
    assert merge_crop_results([confirmation], [validate_crop_result(confirmation, wire())]) == []
    unreadable = wire(assessment="unreadable_text", unreadable=[dict(bbox=[10, 10, 900, 900], reason="blurred")])
    assert merge_crop_results([t], [validate_crop_result(t, unreadable)]) == []
    for value in (wire(complete=False), wire(assessment="uncertain"),
                  wire(assessment="unreadable_text"),
                  wire(unreadable=[dict(bbox=[10, 10, 900, 900], reason="blurred")]),
                  wire(assessment="unreadable_text", unreadable=[dict(bbox=[10, 10, 900, 900], reason="uncertain")])):
        with pytest.raises(OCRRecoveryNeedsReview):
            validate_crop_result(t, value)
    with pytest.raises(OCRRecoveryNeedsReview):
        validate_crop_result(t, wire(), finish_reason="length")


def test_missing_crop_and_inconsistent_checkpoints_fail_closed():
    t = task()
    v = validate_crop_result(t, wire([([10, 10, 900, 50], "keep literal text")]))
    assert CropTask.from_payload(t.to_payload()) == t
    assert ValidatedCrop.from_payload(t, v.to_payload()) == v
    corrupted = deepcopy(v.to_payload()); corrupted["crop_id"] = "other"
    with pytest.raises(OCRRecoveryNeedsReview):
        ValidatedCrop.from_payload(t, corrupted)
    with pytest.raises(OCRRecoveryNeedsReview):
        merge_crop_results([t], [])


def test_numeric_run_cannot_expand_beyond_one_physical_row_capacity():
    t = task()
    with pytest.raises(OCRRecoveryNeedsReview):
        validate_crop_result(t, wire([([1, 1, 100, 20], "1 " + "0 " * 200)]))
    partial = '{"lines":[{"bbox":[1,1,100,20],"text":"' + "0 " * 200
    with pytest.raises(OCRRecoveryNeedsReview):
        stream_review_callback(t)(partial)


def test_length_refinement_retains_full_width_and_all_owned_area():
    parent = task()
    a, b = split_crop_task(parent)
    assert a.source_box[0::2] == b.source_box[0::2] == (0, 1000)
    assert a.ownership_box[1] == 0 and b.ownership_box[3] == 1000
    assert a.ownership_box[3] == b.ownership_box[1]
    assert a.source_box[3] > b.source_box[1]
    assert "owned" in crop_messages("data:image/png;base64,owned", a)[1]["content"][0]["text"]


def test_schema_and_prompts_have_physical_constraints_not_old_regions_text():
    assert localization_response_format()["json_schema"]["strict"] is True
    assert crop_response_format()["json_schema"]["strict"] is True
    assert "complete calendar" in localization_messages("owned")[0]["content"]
    assert "Do not infer missing" in crop_messages("owned", task())[0]["content"]


def test_shifted_overlap_boxes_must_not_be_silently_kept_as_two_rows():
    a = CropTask("a", (1000, 1000), (0, 0, 1000, 800), (0, 0, 1000, 800),
                 (0, 0, 1000, 1000), ("page",))
    b = CropTask("b", (1000, 1000), (0, 200, 1000, 1000), (0, 200, 1000, 1000),
                 (0, 0, 1000, 1000), ("page",))
    va = validate_crop_result(a, wire([([10, 400, 900, 500], "0 0 0")]))
    shifted = validate_crop_result(b, wire([([40, 180, 930, 280], "0 0 0")]))
    with pytest.raises(OCRRecoveryNeedsReview, match="ambiguous"):
        merge_crop_results([a, b], [va, shifted])
    separate = validate_crop_result(b, wire([([10, 400, 900, 450], "0 0 0")]))
    assert merge_crop_results([a, b], [va, separate]) == ["0 0 0", "0 0 0"]


def test_stream_every_character_delta_allows_empty_whitespace_and_half_escapes():
    t = task()
    # Real cumulative deltas stop immediately after the opening quote, spaces,
    # backslashes, escaped quotes and part of an ASCII unicode escape.
    literal = '  literal "quoted" C:\\owned\\新路径 and 中文 42  '
    content = json.dumps(wire([([10, 10, 900, 50], literal)]), ensure_ascii=True)
    guard = stream_review_callback(t)
    for length in range(len(content) + 1):
        guard(content[:length])
    assert validate_crop_result(t, json.loads(content)).lines[0].text == literal


def test_stream_partial_whitespace_waits_but_completed_empty_line_is_rejected():
    guard = stream_review_callback(task())
    prefix = '{"lines":[{"bbox":[10,10,900,50],"text":"'
    for partial in ('', ' ', '   ', '\\', '\\u', '\\u4e', 'literal\\', 'literal\\u4'):
        guard(prefix + partial)
    for completed in ('', ' ', '   '):
        with pytest.raises(OCRRecoveryNeedsReview):
            guard(prefix + completed + '"}')


def test_partial_fix_keeps_physical_capacity_and_duplicate_row_abort():
    t = task()
    guard = stream_review_callback(t)
    # The empty prefix is permitted, then a genuinely over-capacity digit
    # prefix still stops generation without accepting or truncating its text.
    prefix = '{"lines":[{"bbox":[1,1,100,20],"text":"'
    guard(prefix)
    with pytest.raises(OCRRecoveryNeedsReview):
        guard(prefix + '0 ' * 200)
    first = json.dumps(dict(bbox=[10, 10, 900, 50], text='real row'))
    with pytest.raises(OCRRecoveryNeedsReview):
        guard('{"lines":[' + first + ',' + first)


def test_parent_text_region_can_have_completed_blank_owned_bands():
    tasks = plan_crops((1000, 2400), localized([0, 0, 1000, 1000]),
                       max_crop_height=1000, overlap=100, margin=0)
    assert len(tasks) == 3 and all(t.expect_text and t.source_box != t.region_box for t in tasks)
    first = validate_crop_result(tasks[0], wire([([10, 50, 900, 90], "real row in parent")]))
    blanks = [validate_crop_result(t, wire()) for t in tasks[1:]]
    assert merge_crop_results(tasks, [first, *blanks]) == ["real row in parent"]
    assert "not necessarily in this owned horizontal band" in crop_messages("owned", tasks[1])[1]["content"][0]["text"]


def test_every_band_no_text_cannot_erase_a_localized_parent_region():
    tasks = plan_crops((1000, 2400), localized([0, 0, 1000, 1000]),
                       max_crop_height=1000, overlap=100, margin=0)
    blanks = [validate_crop_result(t, wire()) for t in tasks]
    with pytest.raises(OCRRecoveryNeedsReview, match="all bands contradict"):
        merge_crop_results(tasks, blanks)
    unreadable = wire(assessment="unreadable_text",
                      unreadable=[dict(bbox=[10, 10, 900, 200], reason="blurred")])
    explicit = validate_crop_result(tasks[0], unreadable)
    assert merge_crop_results(tasks, [explicit, *blanks[1:]]) == []


def test_a_readable_parent_does_not_mask_an_unresolved_all_blank_other_parent():
    tasks = plan_crops((1000, 2400), localized([0, 0, 400, 1000], [600, 0, 1000, 1000]),
                       max_crop_height=1000, overlap=100, margin=0)
    values = [validate_crop_result(t, wire()) for t in tasks]
    values[0] = validate_crop_result(tasks[0], wire([([10, 50, 900, 90], "one parent has text")]))
    with pytest.raises(OCRRecoveryNeedsReview, match="all bands contradict"):
        merge_crop_results(tasks, values)
