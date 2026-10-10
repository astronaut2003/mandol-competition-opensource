import numpy as np
import pytest

from mandol_competition.code.policies import CodeProcessor,retrieval_profile,PUBLICATION
from mandol_competition.code.api.remote_embeddings import RecipeDenseEncoder,EmbeddingError
from mandol_competition.inference.errors import InferenceIdentityError
from mandol_substrate.state import ProfileMismatch


def test_code_named_model_has_distinct_profile_and_unchanged_optional_policy():
    processor=CodeProcessor()
    original=retrieval_profile(processor)
    assert original.hash==retrieval_profile(processor,'vllm-http-v1').hash
    eligible=retrieval_profile(processor,'aml-official-eligible-v1')
    assert eligible.hash!=original.hash
    assert eligible.embedding_model=='text-embedding-v4' and eligible.embedding_dimension==1024
    assert eligible.splade_model=='disabled'
    assert eligible.options['ranking']==original.options['ranking']
    assert PUBLICATION==__import__('mandol_substrate.contracts',fromlist=['PublicationPolicy']).PublicationPolicy(
        'code-canonical-lexical-optional-dense-v1',False,False)


def test_native_code_encoder_keeps_explicit_query_document_modes():
    class Client:
        modes=[]
        def validate_profile(self):return True
        def embed(self,texts,*,query):
            self.modes.append(query);return [[1.]+[0.]*1023 for _ in texts]
        def close(self):pass
    client=Client();encoder=RecipeDenseEncoder('http://unused',client=client)
    assert encoder.encode([' exact source ']).shape==(1,1024)
    assert encoder.encode_query('query').dtype==np.float32
    assert client.modes==[False,True]


@pytest.mark.parametrize('values',[[1.]*1023,[float('nan')]+[0.]*1023,[0.]*1024,[2.]+[0.]*1023])
def test_native_code_rejects_malformed_dense_before_projection(values):
    class Client:
        def validate_profile(self):return True
        def embed(self,*args,**kwargs):return [values]
    encoder=RecipeDenseEncoder('http://unused',client=Client())
    with pytest.raises(EmbeddingError):encoder.encode(['source'])


def test_native_model_recipe_mismatch_is_not_optional_network_fallback():
    class Client:
        def validate_profile(self):raise InferenceIdentityError('wrong declared recipe')
    encoder=RecipeDenseEncoder('http://unused',client=Client())
    with pytest.raises(ProfileMismatch):encoder.encode(['source'])


def test_native_api_collection_guard_accepts_only_native_code_collection(tmp_path):
    from mandol_competition.code.api.app import create_app
    from mandol_competition.code.api.config import Settings
    native=Settings(inference_contract='aml-official-eligible-v1',coding_db_path=str(tmp_path/'state.sqlite3'),
        qdrant_collection='code_aml_v4_v1')
    assert create_app(native).state.memory_service.store.profile.embedding_model=='text-embedding-v4'
    native.qdrant_collection='textual_aml_v4_v1'
    with pytest.raises(ValueError,match='must be code_aml_v4_v1'):create_app(native)
