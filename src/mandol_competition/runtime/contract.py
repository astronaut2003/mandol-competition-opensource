"""Wire contract boundary: validate internal receipts/results before HTTP 200."""
from dataclasses import dataclass


# Desired output count is independent of the admitted evidence execution budget.
# Formal top_k=100 and all previously admitted 1..100 calls stay unchanged.
EXECUTION_TOP_K_LIMIT=100


class OutputContractError(RuntimeError):
    """An implementation invariant failed; never classify it as client input."""


@dataclass(frozen=True)
class SearchPlan:
    requested_top_k: int
    execution_limit: int | None = None

    @property
    def execution_top_k(self):
        return min(self.requested_top_k,self.execution_limit) if self.execution_limit else self.requested_top_k

    def response(self,hits,model):
        if not isinstance(hits,list) or len(hits)>self.execution_top_k:
            raise OutputContractError('search result count or container is invalid')
        if any(not isinstance(hit,dict) for hit in hits):
            raise OutputContractError('search result item is invalid')
        return model(data=hits)


def acknowledge_add(payload,count,model):
    if type(count) is not int or count!=len(payload.messages):
        raise OutputContractError('Add completion receipt is invalid')
    return model(success=True,request_id=payload.request_id,user_id=payload.user_id,
        session_id=payload.session_id)


def validate_add_response(payload,value,model):
    names=('request_id','user_id','session_id')
    if not isinstance(value,dict) or value.get('success') is not True or any(
            value.get(name)!=getattr(payload,name) for name in names):
        raise OutputContractError('Add completion response is invalid')
    return model(success=True,**{name:value[name] for name in names})


def validate_search_response(value,plan,model):
    if not isinstance(value,dict) or 'data' not in value:
        raise OutputContractError('Search response envelope is invalid')
    return plan.response(value['data'],model)
