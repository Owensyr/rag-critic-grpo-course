"""Conservative, full-sentence rules. No case IDs, answer-key entities or substring credit.

These rules only translate an explicit binary country relation to question-relative
yes/no. They do not infer countries from names or use external world knowledge.
Unsupported clauses, tense mismatches, conflicting answers and uncertain statements
fall back to the semantic verifier, never silently become incorrect.
"""
import re


def clean(text):
    return ' '.join(text.casefold().strip().split())


def country_relation_answer(question, answer):
    q=clean(question)
    if 'not both' in q:
        return None
    m=re.fullmatch(r'(are|were) (.+?) (not )?(?:both )?in '
                   r'(the same country|different countries|two different countries)\?', q)
    if not m:
        return None
    tense,subjects,negative,relation=m.groups()
    entities=subjects.split(' and ')
    # A tightly scoped two-entity question; compound or qualified subjects are unsupported.
    if len(entities)!=2 or any(not re.fullmatch(r"[a-z0-9][a-z0-9 .'-]{0,99}",e) for e in entities):
        return None
    if any(re.search(r'\b(if|not|either|or|only|except|maybe|possibly|whether|who|which|that)\b',e) for e in entities):
        return None
    a=clean(answer)
    # Whole reply only. No trailing explanations, alternatives, quotes or retractions.
    if a.endswith('.'):
        a=a[:-1]
    am=re.fullmatch(r'(.+?) (are|were) (both )?(not )?in '
                    r'(the same country|one country|different countries|two different countries)',a)
    if not am:
        return None
    subject,answer_tense,both,answer_negative,answer_relation=am.groups()
    allowed={subjects,' and '.join(reversed(entities)),'they','both','both of them',
             'the two entities','the two locations'}
    if subject not in allowed or answer_tense!=tense:
        return None
    # "They are not both in ..." has different scope; do not normalize it here.
    if answer_negative and (both or subject in {'both','both of them'}):
        return None
    target_same=relation=='the same country'
    if negative:
        target_same=not target_same
    actual_same=answer_relation in {'the same country','one country'}
    if answer_negative:
        actual_same=not actual_same
    return 'yes' if target_same==actual_same else 'no'


def country_relation_decision(question, answer, gold):
    references={clean(g).rstrip('.') for g in gold}
    if len(references)!=1 or not references.issubset({'yes','no'}):
        return None
    chosen=country_relation_answer(question,answer)
    if chosen is None:
        return None
    return chosen==next(iter(references))


def unsupported_name_suffix(question, answer, gold):
    """An extra generational suffix alone does not prove same/different identity.

    When the accepted list explicitly contains that full name, ordinary exact matching
    remains available. Otherwise this narrow ambiguity is excluded from RL updates.
    """
    if not re.search(r'\b(name|who)\b',clean(question)):
        return False
    def split(value):
        value=clean(value).rstrip('.')
        m=re.fullmatch(r"([a-z][a-z .'-]+?) (jr|sr|ii|iii|iv)\.?",value)
        return (m.group(1),m.group(2)) if m else (value,None)
    a,sa=split(answer)
    refs=[split(g) for g in gold]
    if (a,sa) in refs:
        return False
    return any(a==g and bool(sa)!=bool(sg) for g,sg in refs)
