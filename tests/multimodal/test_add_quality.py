import pytest
from types import SimpleNamespace
from mandol_competition.multimodal.add_quality import annotate,aggregate,KEY

def test_separate_rates_do_not_mix_replays_or_old_unclassified_memories():
    unit=SimpleNamespace(metadata={})
    annotate(unit,['complete','incomplete_ocr','source_only','unverified'])
    result=aggregate([unit.metadata,{},dict(legacy='untagged')])
    assert result['published_images']==4 and result['unclassified_legacy_memories']==2
    assert result['source_only_rate']==result['incomplete_ocr_rate']==.25
    assert result['degradation_rate']==.5 and result['unverified_rate']==.25

def test_empty_population_has_no_fabricated_zero_rate():
    result=aggregate([{}])
    assert result['published_images']==0 and result['source_only_rate'] is None

def test_invalid_quality_cannot_be_silently_counted_as_success():
    with pytest.raises(ValueError):aggregate([{KEY:dict(policy='unknown',images=[])}])
