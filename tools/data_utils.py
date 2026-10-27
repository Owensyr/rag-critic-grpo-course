"""Standard-library data validation for new experiments (no historical asset pack)."""
import json
import math
import re
import unicodedata
from pathlib import Path

NAMES = ('2WikiMultiHopQA', 'ASQA', 'NaturalQuestions', 'PopQA', 'TriviaQA')


def question_key(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', unicodedata.normalize('NFKC', text).casefold()).split())


def stream_array(path, chunk_size=1024 * 1024):
    """Read large JSON arrays without holding a complete author dataset in memory."""
    decoder = json.JSONDecoder()
    with Path(path).open(encoding='utf-8-sig') as f:
        buf = ''; eof = False

        def more():
            nonlocal buf, eof
            part = f.read(chunk_size)
            eof = not part
            buf += part

        def space():
            nonlocal buf
            while True:
                buf = buf.lstrip()
                if buf or eof:
                    return
                more()

        space()
        if not buf.startswith('['):
            raise ValueError('Expected a JSON array: ' + str(path))
        buf = buf[1:]; first = True
        while True:
            space()
            if buf.startswith(']'):
                if (buf[1:] + f.read()).strip():
                    raise ValueError('Trailing data: ' + str(path))
                return
            if not first:
                if not buf.startswith(','):
                    raise ValueError('Missing comma or closing bracket: ' + str(path))
                buf = buf[1:]; space()
                if buf.startswith(']'):
                    raise ValueError('Trailing comma: ' + str(path))
            while True:
                try:
                    row, end = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    more()
            # Dataset entries must be objects, so scalar chunk-boundary ambiguity
            # (e.g. reading 12 as 1) cannot affect this parser.
            if not isinstance(row, dict):
                raise ValueError('Dataset entry must be an object: ' + str(path))
            yield row
            buf = buf[end:]; first = False


def clean_docs(row):
    docs = row['ctxs'][:5]
    if len(docs) != 5:
        raise ValueError('Expected at least five retrieved documents')
    result = []
    for d in docs:
        if not isinstance(d['title'], str) or not isinstance(d['text'], str) or not d['text'].strip():
            raise ValueError('Invalid retrieved document')
        score = float(d['score']) if d.get('score') is not None else None
        if score is not None and not math.isfinite(score):
            raise ValueError('Nonfinite retrieval score')
        result.append(dict(title=d['title'], text=d['text'], score=score))
    return result


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]
