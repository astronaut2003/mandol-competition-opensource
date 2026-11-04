"""Contention before GC is distinct from a failed GC operation."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from mandol_competition.textual.api.competition import CompetitionMemoryService
from mandol_substrate.state import Scope,StateUnavailable
from mandol_substrate.state.models import ScopeBusy


class OnePass:
    count=0
    def wait(self,seconds):self.count+=1;return self.count>1
    def is_set(self):return False


@pytest.mark.parametrize('prior',[None,'earlier real GC failure'])
def test_busy_scope_is_skipped_without_poisoning_health_or_hiding_prior_failure(prior):
    busy,free=Scope('artificial','busy'),Scope('artificial','free');collected=[]
    @contextmanager
    def view(scope):
        if scope==busy:raise ScopeBusy('artificial contention')
        yield
    service=CompetitionMemoryService.__new__(CompetitionMemoryService)
    service._maintenance_stop=OnePass();service._maintenance_error=prior
    service.state=SimpleNamespace(iter_scopes=lambda:iter([busy,free]),
        scope_state=lambda scope:dict(canonical_seq=1,published_seq=1))
    service.run=SimpleNamespace(maintenance_view=view)
    service.projection=SimpleNamespace(gc_scope=collected.append)
    service._maintain()
    assert collected==[free] and service._maintenance_error==prior


@pytest.mark.parametrize('failure',[StateUnavailable,ScopeBusy])
def test_actual_gc_failure_still_marks_maintenance_unavailable(failure):
    scope=Scope('artificial','user')
    @contextmanager
    def view(scope):yield
    def fail(scope):raise failure('artificial GC outage')
    service=CompetitionMemoryService.__new__(CompetitionMemoryService)
    service._maintenance_stop=OnePass();service._maintenance_error=None
    service.state=SimpleNamespace(iter_scopes=lambda:iter([scope]),
        scope_state=lambda scope:dict(canonical_seq=1,published_seq=1))
    service.run=SimpleNamespace(maintenance_view=view)
    service.projection=SimpleNamespace(gc_scope=fail)
    service._maintain()
    assert service._maintenance_error=='projection maintenance failed: '+failure.__name__
