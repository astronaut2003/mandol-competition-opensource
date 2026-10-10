import os
import uuid
import numpy as np
from mandol_competition.code.api.coding_memory import CodingMemoryService
from mandol_competition.code.api.config import Settings
from mandol_competition.code.projection import prepare_channel


def test_code_optional_dense_exact_order_and_unavailable_fallback(tmp_path):
    settings = Settings(coding_db_path=str(tmp_path / 'state.sqlite3'), qdrant_endpoint=os.environ['MANDOL_TEST_QDRANT'],
                        qdrant_collection='code_' + uuid.uuid4().hex, coding_retrieval_mode='raw')
    service = CodingMemoryService(settings); service.start()
    class Encoder:
        def encode(self, texts):
            values = []
            for text in texts:
                v = np.zeros(1024, dtype=np.float32)
                v[0], v[1] = 1., (sum(text.encode()) % 101) / 101
                v /= np.linalg.norm(v.astype(np.float64))
                values.append(v)
            return np.array(values)
    try:
        service._encoder = Encoder()
        for i in range(3):
            service.add_messages('u', f's{i}', f'r{i}', [{'role': 'user', 'content': 'alpha ' + str(i)}])
        scope = service.store.scope('u')
        assert not service.store.state.projection_ready(scope, 'dense')
        memories = {m.logical_uid: m.text_content for batch in service.store.state.iter_memories(scope) for m in batch}
        channel = prepare_channel(service, 'u', memories)
        expected_vectors = service._encoder.encode([memories[u] for u in sorted(memories)])
        assert np.array_equal(channel.vectors, expected_vectors)
        expected = sorted(zip(sorted(memories), map(float, expected_vectors @ service._encoder.encode(['alpha'])[0])), key=lambda x: (-x[1], x[0]))
        assert channel.search('alpha', top_k=100) == expected
        assert service.search('u', 'alpha', 100)
        service._vector.client.delete_collection(service._vector.collection_name)
        # Existing canonical acknowledgement survives optional channel loss.
        assert service.search('u', 'alpha', 100)
        assert service._last_retrieval_path == 'LEXICAL_FALLBACK'
        assert service.store.state._rows('SELECT * FROM runtime_events')
    finally:
        if service._vector:
            service._vector.client.delete_collection(service._vector.collection_name)
        service.shutdown()
