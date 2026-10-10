from mandol_competition.inference.clients import HTTPSparseEncoderClient
from mandol_competition.inference.dashscope import official_embedding_recipes


def test_standalone_sparse_query_does_not_silently_use_provider_document_mode():
    class Transport:
        bodies=[]
        def request(self,method,path,body):
            self.bodies.append(body)
            return {'model':'text-embedding-v4','data':[{'index':0,'sparse':{'51765':.8}}]}
    transport=Transport()
    client=HTTPSparseEncoderClient('http://unused',official_embedding_recipes()[1],transport=transport)
    assert client.encode_documents(['source'])==[{51765:.8}]
    assert client.encode_query('question')=={51765:.8}
    assert 'input_type' not in transport.bodies[0]  # existing document wire unchanged
    assert transport.bodies[1]['input_type']=='query'
