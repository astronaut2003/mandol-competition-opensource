from mandol_competition.multimodal.render import Location, RenderImage, render_query_text, render_text_content
from mandol_competition.multimodal.vlm.base import Entity, Event, Extraction, Relation, VisualObject


def _image(media_id="img_1", **overrides):
    payload = dict(
        description="A red backpack on a car seat.",
        entities=[Entity("Grace", "person")],
        objects=[VisualObject("red backpack"), VisualObject("car")],
        ocr_text="SALE 50%",
        events=[Event("Grace", "holding", "red backpack")],
        relations=[Relation("Grace", "next_to", "car")],
        tags=["outdoor", "travel"],
    )
    payload.update(overrides)
    return RenderImage(media_id, Extraction(**payload))


def test_memory_render_is_deterministic_and_complete():
    args = dict(dialogue="Can it play a role?", timestamp_text="2024-09-10", images=[_image()])
    expected = [
        "[Dialogue] Can it play a role?",
        "[Timestamp] 2024-09-10",
        "[Image img_1]",
        "Visual description: A red backpack on a car seat.",
        "Entities: Grace (person)",
        "Objects: red backpack; car",
        "OCR: SALE 50%",
        "Events: Grace holding red backpack",
        "Relations: Grace next_to car",
        "Tags: outdoor; travel",
    ]
    assert render_text_content(**args).splitlines() == expected
    assert render_text_content(**args) == render_text_content(**args)


def test_dialogue_preserves_original_text_and_order():
    source = "D1:IMG_002  first\n second \t标识符  "
    assert render_text_content(dialogue=source, timestamp_text=None, images=[]) == "[Dialogue] " + source


def test_query_preserves_question_and_options_original_text():
    result = render_query_text(query_text="A  B\nC", images=[], options=[" A.   yes ", "B. no"])
    assert result == "[Question] A  B\nC\n[Options]\n A.   yes \nB. no"


def test_empty_extraction_and_image_only_skip_empty_lines():
    image = _image(description="", entities=[], objects=[], ocr_text="", events=[], relations=[], tags=[])
    assert render_text_content(dialogue="", timestamp_text=None, images=[image]) == "[Image img_1]"
    assert render_query_text(query_text="", images=[image]) == "[Image img_1]"


def test_image_blocks_keep_input_order_and_query_label_parity():
    images = [_image("img_1"), _image("img_2")]
    memory = render_text_content(dialogue="", timestamp_text=None, images=images)
    query = render_query_text(query_text="what?", images=images)
    assert memory == query.removeprefix("[Question] what?\n")
    assert memory.index("[Image img_1]") < memory.index("[Image img_2]")


def test_structured_multiline_values_are_collapsed_and_count_kept():
    image = _image(description="a\nred bag", entities=[Entity("Grace", "person", [" tall\nadult "])],
                   objects=[VisualObject("balloon", 3, ["bright\nblue"])], ocr_text="OPEN\n9-5\nMon")
    text = render_text_content(dialogue="", timestamp_text=None, images=[image])
    assert "Visual description: a red bag" in text
    assert "Entities: Grace (person; tall adult)" in text
    assert "Objects: balloon (x3; bright blue)" in text
    assert "OCR: OPEN 9-5 Mon" in text


def test_location_requires_explicit_provenance_not_vision_guess():
    image = _image(location_hint="hospital")
    assert "Location:" not in render_text_content(dialogue="", timestamp_text=None, images=[image])
    guessed = RenderImage(image.media_id, image.extraction, Location("hospital", "vlm_hint"))
    assert "Location:" not in render_text_content(dialogue="", timestamp_text=None, images=[guessed])
    located = RenderImage(image.media_id, image.extraction, Location("50.906,-1.418", "exif"))
    assert "Location: 50.906,-1.418 (source: exif)" in render_text_content(
        dialogue="", timestamp_text=None, images=[located]
    )
