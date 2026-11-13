"""Pure durable visual-stage accounting; no provider, filesystem or database.

``spent`` is a conservative charge within one explicit policy grant, not a
claim about historical provider invoices. Unknown dispatches consume their
reserved ceiling. Legacy state explicitly carries unknown earlier costs.
"""
from copy import deepcopy
import hashlib
import json
import math
import re

FORMAT = 'vision-stage-state-v1'
MAX_OUTPUT_TOKENS = 16384
MAX_PAID_CALLS = 20
MAX_CONTENT_FAILURES = 4
MAX_JSON_BYTES = 2 * 1024 * 1024
REVIEW_REASONS = frozenset({'truncated_ocr', 'invalid_ocr', 'sequential_numbers', 'repeated_lines',
                           'numeric_repetition', 'long_output'})
UPDATE_FIELDS = frozenset({'scene', 'localization', 'tasks', 'results', 'coverage',
                           'complete_receipt', 'final_payload', 'final_usage', 'ocr_repair', 'scene_recovery', 'ocr_strategy'})
STATE_FIELDS = frozenset({'format', 'binding', 'binding_sha256', 'spent', 'observed_tokens',
    'paid_calls', 'content_failures', 'known_rejections', 'reservation', 'calls',
    'historical_cost_unknown', 'single_new_policy_grant', 'final_payload_sha256', *UPDATE_FIELDS})


class RecoveryStateError(RuntimeError):
    pass


class RecoveryBudgetExhausted(RecoveryStateError):
    pass


def _need(value, reason):
    if not value:
        raise RecoveryStateError(reason)


def _json_copy(value):
    nodes = 0
    def visit(item, depth):
        nonlocal nodes
        nodes += 1
        _need(nodes <= 40000 and depth <= 32, 'stage checkpoint nesting/node bound')
        if item is None or type(item) in (str, bool, int, float):
            if type(item) is float:
                _need(math.isfinite(item), 'stage checkpoint requires finite JSON')
            return
        if type(item) is dict:
            _need(len(item) <= 4096 and all(type(k) is str for k in item), 'stage checkpoint object bound/type')
            for child in item.values():
                visit(child, depth + 1)
            return
        if type(item) is list:
            _need(len(item) <= 4096, 'stage checkpoint list bound')
            for child in item:
                visit(child, depth + 1)
            return
        raise RecoveryStateError('stage checkpoint requires plain JSON values')
    visit(value, 0)
    try:
        raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        raise RecoveryStateError('stage checkpoint JSON encoding failed') from exc
    _need(len(raw) <= MAX_JSON_BYTES, 'stage checkpoint byte bound')
    return json.loads(raw)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _binding_has_image(value):
    if type(value) is str:
        return re.match(r'^data:image/[^;,\s]+;base64,', value) is not None
    if type(value) is dict:
        return any(k in {'messages', 'image_data_uri', 'image_url', 'image_data'}
                   or _binding_has_image(v) for k, v in value.items())
    return type(value) is list and any(_binding_has_image(v) for v in value)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _stage(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:-]{1,80}', value) is not None


def _metadata(value, *, limit=None):
    _need(type(value) is dict, 'stage response metadata must be object')
    budget = value.get('output_token_budget')
    completion = value.get('completion_tokens')
    _need(_integer(budget, 1, MAX_OUTPUT_TOKENS) and (limit is None or budget == limit), 'stage response budget differs')
    _need(completion is None or _integer(completion, 0, budget), 'stage completion tokens outside budget')
    reason = value.get('finish_reason')
    _need(reason is None or type(reason) is str and len(reason) <= 128, 'stage finish reason type')
    return completion


def _usage(value, completion=None):
    _need(value is None or type(value) is dict, 'stage usage must be object or null')
    if value is not None:
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if key in value:
                _need(_integer(value[key], 0, 2**53 - 1), 'stage usage token type/range')
        other = value.get('completion_tokens')
        _need(completion is None or other is None or completion == other, 'stage usage completion mismatch')
    return value.get('completion_tokens') if value is not None else None


def _scene(value):
    if value is None:
        return
    _need(type(value) is dict and set(value) == {'payload', 'payload_sha256', 'metadata', 'usage', 'review_reason'},
          'scene checkpoint shape')
    from .ocr_lines import VisualScene
    try:
        parsed = VisualScene.model_validate(value['payload']).model_dump()
    except (TypeError, ValueError) as exc:
        raise RecoveryStateError('scene checkpoint schema') from exc
    _need(parsed == value['payload'] and digest(parsed) == value['payload_sha256'], 'scene checkpoint SHA differs')
    completion = _metadata(value['metadata'])
    _usage(value['usage'], completion)
    _need(value['metadata'].get('finish_reason') in {'stop', 'length'}
          and type(value['review_reason']) is str and value['review_reason'] in REVIEW_REASONS,
          'scene checkpoint response/review reason')


def _validate(state, binding):
    _need(type(state) is dict and set(state) == STATE_FIELDS and state.get('format') == FORMAT, 'stage checkpoint shape/version')
    _need(type(state['binding']) is dict and state['binding'] == binding
          and state['binding_sha256'] == digest(binding), 'stage checkpoint binding differs')
    for key, maximum in (('spent', MAX_OUTPUT_TOKENS), ('observed_tokens', MAX_OUTPUT_TOKENS),
                         ('paid_calls', MAX_PAID_CALLS), ('content_failures', MAX_CONTENT_FAILURES),
                         ('known_rejections', 2**53 - 1)):
        _need(_integer(state[key], 0, maximum), 'stage checkpoint counter range/type')
    _need(type(state['historical_cost_unknown']) is bool and type(state['single_new_policy_grant']) is bool
          and state['historical_cost_unknown'] == state['single_new_policy_grant'], 'legacy policy grant marker')
    _need(type(state['calls']) is list and len(state['calls']) <= MAX_PAID_CALLS, 'stage call history bound')
    charged = observed = paid = rejected = 0
    for call in state['calls']:
        _need(type(call) is dict and set(call) == {'stage', 'limit', 'metadata', 'usage', 'charged',
              'observed_tokens', 'known_rejection'}, 'stage call history shape')
        _need(_stage(call['stage']) and _integer(call['limit'], 1, MAX_OUTPUT_TOKENS)
              and type(call['metadata']) is dict and (call['usage'] is None or type(call['usage']) is dict)
              and type(call['known_rejection']) is bool, 'stage call history types')
        _need(_integer(call['charged'], 0, call['limit'])
              and _integer(call['observed_tokens'], 0, call['charged']), 'stage call history charge')
        if call['known_rejection']:
            _need(call['charged'] == call['observed_tokens'] == 0, 'known rejection must be free')
            rejected += 1
        else:
            paid += 1
        charged += call['charged']; observed += call['observed_tokens']
    reservation = state['reservation']
    if reservation is not None:
        _need(type(reservation) is dict and set(reservation) == {'stage', 'limit'}
              and _stage(reservation['stage']) and _integer(reservation['limit'], 1, MAX_OUTPUT_TOKENS),
              'stage reservation shape')
        charged += reservation['limit']; paid += 1
    _need(charged == state['spent'] and observed == state['observed_tokens'] and paid == state['paid_calls']
          and rejected <= state['known_rejections'], 'stage checkpoint accounting mismatch')
    _scene(state['scene'])
    from . import ocr_strategy
    try:
        ocr_strategy.validate(state['ocr_strategy'],state['calls'])
    except (TypeError,ValueError,KeyError) as exc:
        raise RecoveryStateError('OCR strategy checkpoint invalid') from exc
    from . import ocr_concise
    _need(len(ocr_concise.paid_calls(state['calls']))<=ocr_concise.MAX_CALLS,
          'concise OCR allowance exceeded')
    from . import scene_recovery
    try:
        scene_recovery.validate_checkpoint(state['scene_recovery'], state['calls'])
        if state['scene'] is not None and state['scene']['metadata'].get('scene_recovery_policy') is not None:
            _need(state['scene_recovery'] is not None
                  and scene_recovery.scene_record(state['scene_recovery']) == state['scene'],
                  'scene recovery aggregate differs from completed pages')
    except (TypeError, ValueError) as exc:
        raise RecoveryStateError('scene recovery checkpoint invalid') from exc
    repair = state['ocr_repair']
    if repair is not None:
        from . import ocr_review
        _need(type(repair) is dict and set(repair) == {'policy', 'reason', 'failures', 'review'},
              'OCR repair checkpoint shape')
        _need(repair['policy'] == ocr_review.policy() and type(repair['reason']) is str
              and repair['reason'] in ocr_review.REASONS
              and _integer(repair['failures'], 0, ocr_review.MAX_REPAIR_FAILURES), 'OCR repair policy/budget')
        review = repair['review']
        if review is not None:
            _need(type(review) is dict and set(review) == {'wire', 'metadata', 'usage'}, 'OCR review shape')
            try:
                parsed = ocr_review.ReviewWire.model_validate(review['wire']).model_dump()
            except (ValueError, TypeError) as exc:
                raise RecoveryStateError('OCR review schema') from exc
            _need(parsed == review['wire'] and parsed['inspected_entire_image']
                  and parsed['assessment'] != 'uncertain', 'OCR review coverage')
            tokens = _metadata(review['metadata'])
            _usage(review['usage'], tokens)
            _need(review['metadata']['finish_reason'] == 'stop'
                  and (tokens is None or tokens < review['metadata']['output_token_budget']),
                  'OCR review must be complete')
    _need(type(state['tasks']) is list and type(state['results']) is dict, 'stage task/result types')
    for key in ('localization', 'complete_receipt', 'final_payload', 'final_usage'):
        _need(state[key] is None or type(state[key]) is dict, 'stage optional object type')
    _need(state['final_payload_sha256'] == (digest(state['final_payload'])
          if state['final_payload'] is not None else None), 'completed payload fingerprint differs')


def _known_not_paid(exc):
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen and len(chain) < 8:
        seen.add(id(exc)); chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    if any(getattr(e, 'vision_paid_output_failure', False) or getattr(e, 'paid_output_failure', False)
           or getattr(e, 'unknown_outcome', False) for e in chain):
        return False
    for error in chain:
        known = getattr(error, 'unknown_outcome', None) is False
        if known and (getattr(error, 'status_code', None) == 429 or getattr(error, 'vision_not_dispatched', False)):
            return True
        name = type(error).__name__
        code = getattr(error, 'error_class', None)
        if name in {'ConnectError', 'ConnectTimeout', 'PoolTimeout', 'ProxyError', 'InferenceDeadlineExceeded'}:
            return True
        if known and code in {'ConnectError', 'ConnectTimeout', 'PoolTimeout', 'ProxyError', 'deadline_before_dispatch'}:
            return True
    return False


class StageState:
    def __init__(self, binding, resume=None, checkpoint=None, legacy=False):
        _need(type(binding) is dict and bool(binding), 'nonempty stage binding required')
        _need(checkpoint is None or callable(checkpoint), 'checkpoint callback must be callable')
        _need(type(legacy) is bool, 'legacy flag must be boolean')
        self.binding = _json_copy(binding)
        _need(not _binding_has_image(self.binding), 'stage binding must contain image hashes rather than image payloads')
        self._checkpoint, self._poisoned = checkpoint, False
        if resume is None:
            value = dict(format=FORMAT, binding=deepcopy(self.binding), binding_sha256=digest(self.binding),
                spent=0, observed_tokens=0, paid_calls=0, content_failures=0, known_rejections=0,
                reservation=None, calls=[], historical_cost_unknown=legacy, single_new_policy_grant=legacy,
                scene=None, localization=None, tasks=[], results={}, coverage=None,
                complete_receipt=None, final_payload=None, final_payload_sha256=None, final_usage=None,
                ocr_repair=None, scene_recovery=None, ocr_strategy=None)
            self._state = None
            self._persist(value)  # The one legacy grant exists durably before dispatch.
        else:
            value = _json_copy(resume)
            # Existing v1 checkpoints keep their original binding and all costs.
            # The new optional grant is versioned independently of that binding.
            if type(value) is dict:
                value.setdefault('ocr_repair', None)
                value.setdefault('scene_recovery', None)
                value.setdefault('ocr_strategy', None)
            _validate(value, self.binding)
            self._state = value
            if value['reservation'] is not None:
                saved = deepcopy(value)
                reservation = saved.pop('reservation'); saved['reservation'] = None
                self._append(saved, dict(stage=reservation['stage'], limit=reservation['limit'],
                    metadata={'error_class': 'dispatch_outcome_unknown_after_resume'}, usage=None,
                    charged=reservation['limit'], observed_tokens=0, known_rejection=False))
                # reserve already charged it; recovery must not charge it twice.
                self._persist(saved)

    @property
    def state(self):
        return deepcopy(self._state)

    def _ready(self):
        _need(not self._poisoned, 'checkpoint persistence failed; new dispatch forbidden')

    def _persist(self, value):
        self._ready()
        value = _json_copy(value)
        _validate(value, self.binding)
        try:
            if self._checkpoint is not None:
                self._checkpoint(deepcopy(value))
        except Exception as exc:
            self._poisoned = True
            raise RecoveryStateError('stage checkpoint persistence failed') from exc
        except BaseException:
            self._poisoned = True
            raise
        self._state = value

    @staticmethod
    def _append(value, call):
        # Keep every paid/unknown stage. A long outage may compact only free
        # rejections, whose cumulative count lives separately in the state.
        if len(value['calls']) == MAX_PAID_CALLS:
            removable = next((i for i, item in enumerate(value['calls']) if item['known_rejection']), None)
            _need(removable is not None, 'no free rejection history slot')
            value['calls'].pop(removable)
        value['calls'].append(call)

    def reserve(self, stage, limit):
        self._ready()
        _need(_stage(stage) and _integer(limit, 1, MAX_OUTPUT_TOKENS), 'stage reservation type/range')
        _need(self._state['reservation'] is None, 'stage reservation already active')
        _need(self._state['complete_receipt'] is None, 'completed stage state cannot dispatch again')
        from . import scene_recovery
        if scene_recovery.is_stage(stage) and not scene_recovery.authorized(
                self._state['scene_recovery'], self._state['calls'], stage):
            raise RecoveryBudgetExhausted('scene field recovery allowance exhausted')
        from . import ocr_concise
        if ocr_concise.is_stage(stage) and not ocr_concise.authorized(self._state,stage):
            raise RecoveryBudgetExhausted('concise OCR allowance exhausted')
        from . import ocr_strategy
        if ocr_strategy.is_stage(stage) and not ocr_strategy.authorized(self._state,stage):
            raise RecoveryStateError('unbound OCR strategy dispatch')
        repair = self._state['ocr_repair']
        from . import ocr_review
        repair_stage = (stage in {ocr_review.review_stage(), ocr_review.crop_stage()}
                        or repair is not None and (ocr_concise.is_stage(stage) or ocr_strategy.is_stage(stage)))
        repairing = (repair is not None and repair_stage
                     and repair['failures'] < ocr_review.MAX_REPAIR_FAILURES)
        if stage == ocr_review.review_stage():
            reviews = sum(c['stage'] == stage and not c['known_rejection'] for c in self._state['calls'])
            if reviews >= ocr_review.MAX_REVIEWS:
                raise RecoveryBudgetExhausted('independent OCR review allowance exhausted')
        if (self._state['spent'] + limit > MAX_OUTPUT_TOKENS or self._state['paid_calls'] >= MAX_PAID_CALLS
                or self._state['content_failures'] >= MAX_CONTENT_FAILURES and not repairing
                or repair_stage and not repairing):
            raise RecoveryBudgetExhausted('cumulative visual recovery budget exhausted')
        value = self.state
        value['spent'] += limit; value['paid_calls'] += 1
        value['reservation'] = dict(stage=stage, limit=limit)
        self._persist(value)
        return deepcopy(value['reservation'])

    def accept(self, metadata, usage=None):
        self._ready()
        reservation = self._state['reservation']
        _need(reservation is not None, 'no active stage reservation')
        metadata, usage = _json_copy(metadata), _json_copy(usage)
        observed = _metadata(metadata, limit=reservation['limit'])
        usage_observed = _usage(usage, observed)
        if observed is None:
            observed = usage_observed
        _need(observed is None or _integer(observed, 0, reservation['limit']), 'stage observed output exceeds reservation')
        charged = observed if observed is not None else reservation['limit']
        value = self.state
        value['spent'] -= reservation['limit'] - charged
        value['observed_tokens'] += observed if observed is not None else 0
        value['reservation'] = None
        self._append(value, dict(**reservation, metadata=metadata, usage=usage, charged=charged,
                                observed_tokens=observed if observed is not None else 0, known_rejection=False))
        self._persist(value)
        return deepcopy(dict(stage=reservation['stage'], metadata=metadata, usage=usage))

    def reject(self, exc):
        self._ready()
        reservation = self._state['reservation']
        _need(reservation is not None and isinstance(exc, BaseException), 'rejected stage reservation required')
        value = self.state
        free = _known_not_paid(exc)
        if free:
            value['spent'] -= reservation['limit']; value['paid_calls'] -= 1
            value['known_rejections'] += 1
        observed = getattr(exc, 'vlm_completion_tokens', None)
        observed = observed if not free and _integer(observed, 0, reservation['limit']) else 0
        value['observed_tokens'] += observed
        value['reservation'] = None
        paid_content=bool(getattr(exc,'vision_paid_output_failure',False) or getattr(exc,'paid_output_failure',False))
        code='known_not_dispatched' if free else 'content_output_rejected' if paid_content else 'unverified_paid_outcome'
        self._append(value, dict(**reservation, metadata={'error_class': code},
            usage=None, charged=0 if free else reservation['limit'], observed_tokens=observed, known_rejection=free))
        self._persist(value)

    def content_failure(self):
        self._ready()
        if self._state['content_failures'] >= MAX_CONTENT_FAILURES:
            raise RecoveryBudgetExhausted('cumulative content failure budget exhausted')
        value = self.state; value['content_failures'] += 1
        self._persist(value)

    def repair_failure(self):
        """Separate, one-policy bounded recovery; never reset historical costs."""
        self._ready()
        from . import ocr_review
        value = self.state
        repair = value['ocr_repair']
        _need(repair is not None, 'OCR repair must be durably started')
        if repair['failures'] >= ocr_review.MAX_REPAIR_FAILURES:
            raise RecoveryBudgetExhausted('independent OCR repair failures exhausted')
        repair['failures'] += 1
        self._persist(value)

    def update(self, **fields):
        self._ready()
        _need(bool(fields) and set(fields) <= UPDATE_FIELDS, 'only non-budget stage fields may be updated')
        value = self.state
        if value['scene'] is not None and 'scene' in fields:
            _need(fields['scene'] == value['scene'], 'validated scene cannot be replaced or erased')
        if value['scene_recovery'] is not None and 'scene_recovery' in fields:
            prior, new = value['scene_recovery'], fields['scene_recovery']
            _need(type(new) is dict and new.get('policy') == prior['policy']
                  and type(new.get('pages')) is list
                  and new['pages'][:len(prior['pages'])] == prior['pages'],
                  'completed scene field pages cannot be replaced or erased')
        if value['ocr_repair'] is not None and 'ocr_repair' in fields:
            prior, new = value['ocr_repair'], fields['ocr_repair']
            _need(type(new) is dict and new.get('policy') == prior['policy']
                  and type(new.get('failures')) is int and new['failures'] >= prior['failures'],
                  'OCR repair grant and failures cannot be erased or refreshed')
        if value['final_payload'] is not None and 'final_payload' in fields:
            _need(fields['final_payload'] == value['final_payload'], 'completed payload cannot be replaced or erased')
        if value['ocr_strategy'] is not None and 'ocr_strategy' in fields:
            _need(fields['ocr_strategy']==value['ocr_strategy'],'verified OCR strategy cannot be replaced')
        value.update(deepcopy(fields))
        if 'final_payload' in fields:
            value['final_payload_sha256'] = digest(fields['final_payload']) if fields['final_payload'] is not None else None
        self._persist(value)
