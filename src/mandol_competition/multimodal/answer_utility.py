"""Opt-in output policies over existing evidence; no answer or new memory."""
import re


def adaptive_depth(question, has_image):
    """Generic signals frozen before evaluation, with no dataset/label input."""
    if re.search(r'\b(all|list|before|after|compare|events|steps)\b|所有|哪些|之前|之后|比较|步骤',question,re.I):
        return 16
    return 10 if has_image else 6


def validate_policy(policy):
    if (not isinstance(policy,dict)
            or policy.get('version') not in {'mm-answer-utility-v1','mm-answer-utility-v2'}
            or set(policy)-{'version','candidate_k','output_k','selector','depth'}
            or type(policy.get('candidate_k',30)) is not int
            or not 1<=policy.get('candidate_k',30)<=100
            or type(policy.get('output_k',8)) is not int
            or not 1<=policy.get('output_k',8)<=100
            or policy.get('selector','none') not in {'none','text','visual'}
            or policy.get('depth','fixed') not in {'fixed','adaptive-v1'}
            or policy.get('version')=='mm-answer-utility-v1' and 'depth' in policy):
        raise ValueError('invalid versioned answer-utility policy')
    return policy


def output_limit(policy, question, has_image, requested):
    limit=adaptive_depth(question,has_image) if policy.get('depth')=='adaptive-v1' else policy.get('output_k',requested)
    return min(requested,limit)
