"""Mixed OCR attribution for serving; no full-Extraction claim or ranking changes."""
from mandol_substrate.state import Scope, StateUnavailable
from mandol_substrate.state.models import digest
from .ocr_field_repair import HISTORY_KEY, POLICY, PROVENANCE_POLICY, retained_image_hashes, text_sha

def ocr_source_provenance(unit,*,namespace=None):
    history=unit.metadata.get(HISTORY_KEY)
    if history is None:return None
    try:
        if not isinstance(history,list) or not history:raise ValueError()
        images=unit.raw_data["multimodal_info"]["images"];latest={};chains={}
        for event in history:
            component=event["component"];position=component["image_position"]
            scope=Scope(**event["source_scope"])
            if (event["policy"]!=POLICY or event["provenance_policy"]!=PROVENANCE_POLICY
                    or type(position) is not int or not 0<=position<len(images)
                    or event["source_uid"]!=unit.uid or scope.key!=event["source_scope_key"]
                    or scope.user_id!=unit.metadata["user_id"]
                    or namespace is not None and scope.namespace!=namespace
                    or component["field"]!="ocr_text" or component["image_count"]!=len(images)
                    or event["component_sha256"]!=digest(component)
                    or event["full_extraction_receipt_validates_hybrid"] is not False):
                raise ValueError()
            if position in chains and chains[position]!=component["old_value_sha256"]:
                raise ValueError()
            chains[position]=component["value_sha256"];latest[position]=event
        summary=[]
        for position,event in sorted(latest.items()):
            image=images[position];component=event["component"]
            if (component["value_sha256"]!=text_sha(image["ocr_text"])
                    or component["media_id"]!=image["media_id"]
                    or component["source_context_sha256"]!=text_sha(unit.raw_data["original_content"])
                    or event["retained_image_field_sha256"]!=retained_image_hashes(image)
                    or event["historical_identity"]!={k:image[k] for k in ("model","prompt_version","schema_version")}):
                raise ValueError()
            producer=component["producer"]
            if producer.get("provider")=="local-cpu-offline":
                identity=dict(provider="local-cpu-offline",engine=producer["engine"],models=producer["models"])
            elif producer.get("provider")=="closeai":
                identity={k:producer[k] for k in ("provider","model","prompt_version","model_context_policy")}
            else:raise ValueError()
            summary.append(dict(image_position=position,ocr_identity=identity,
                retained_fields_identity=event["historical_identity"],retained_fields=sorted(event["retained_image_field_sha256"]),
                independent_evidence_sha256=component["evidence_sha256"],
                full_extraction_receipt_validates_hybrid=False))
        return dict(policy=PROVENANCE_POLICY,images=summary)
    except (KeyError,TypeError,ValueError,IndexError) as exc:
        raise StateUnavailable("mixed OCR field provenance is inconsistent") from exc

def incomplete_ocr_note(unit):
    values = unit.metadata.get('visual_ocr_readability', [])
    if not isinstance(values, list):
        return None
    positions = [str(v['image_position']+1) for v in values if isinstance(v, dict)
        and v.get('status') == 'incomplete' and type(v.get('image_position')) is int
        and 0 <= v['image_position'] < 1000]
    if not positions:
        return None
    return ('[OCR extraction status] Image ' + ', '.join(positions) +
        ': OCR is incomplete after bounded recovery. Original image and source text are retained. '
        'Missing OCR is not evidence that the image contains no text. '
        'Only validated text fragments are transcribed; exact untranscribed details remain unverified.')


def ocr_delivery_note(unit,*,namespace=None):
    result=ocr_source_provenance(unit,namespace=namespace)
    incomplete = incomplete_ocr_note(unit)
    if result is None:return incomplete
    positions=", ".join(str(v["image_position"]+1) for v in result["images"])
    note = ("[OCR field provenance] Image "+positions+": OCR text comes from a separate reviewed image transcription. "
        "Description, objects and other image facts retain their historical extraction identity. "
        "A historical full-extraction receipt does not validate the combined image record.")
    return note + ('\n' + incomplete if incomplete else '')

