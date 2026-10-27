import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from common import load,save,sha,configuration
from data_utils import NAMES,stream_array
from prepare_data import prepare_course,select_grpo
from prepare_workdir import prepare,validate_new_selection
from unpack_author_data import unpack


def row(i):
    return dict(id=i,question=f'Which place belongs to unique item {i}?',answers=['Place'],
                rationale='SECRET_RATIONALE',ctxs=[dict(title='Title',text='Passage',score='1.5',has_answer=True) for _ in range(5)])


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.d=Path(self.temp.name)

    def sources(self):
        author=self.d/'author'
        for n in NAMES:save(author/n/'test.json',[row(1)])
        save(author/'2WikiMultiHopQA/train.json',[row(i) for i in range(270)]+[row(269)])
        course=self.d/'test.json';save(course,[dict(row(2),answers=None)])
        sft=self.d/'sft.jsonl';sft.write_text(json.dumps(row(0))+'\n',encoding='utf-8')
        return author,course,sft

    def test_stream_large_chunk_boundaries(self):
        p=self.d/'array.json';save(p,[row(1),row(2)])
        self.assertEqual(list(stream_array(p,7)),[row(1),row(2)])

    def test_stream_rejects_malformed_arrays(self):
        p=self.d/'bad.json'
        for text in ('[{"a":1},]','[{"a":1}', '[{}] garbage', '[1]', '[{} {}]'):
            p.write_text(text,encoding='utf-8')
            with self.assertRaises(ValueError):list(stream_array(p,3))

    def test_course_normalizes_without_gold_or_hidden_metadata(self):
        p=self.d/'test.json';r=row(0);r.pop('answers');save(p,[r])
        sample=self.d/'sample.csv';sample.write_text('id,answer\n0,\n')
        prepare_course(p,sample,self.d/'out',expected_rows=1)
        got=load(self.d/'out/test.json')[0]
        self.assertEqual(set(got),{'id','question','ctxs','answers'})
        self.assertIsNone(got['answers']);self.assertEqual(got['ctxs'][0],dict(title='Title',text='Passage',score=1.5))
        with self.assertRaises(FileExistsError):prepare_course(p,sample,self.d/'out',expected_rows=1)

    def test_course_rejects_labels_and_template_mismatch(self):
        p=self.d/'test.json';save(p,[row(0)])
        sample=self.d/'sample.csv';sample.write_text('id,answer\n0,\n')
        with self.assertRaises(ValueError):prepare_course(p,sample,self.d/'bad',expected_rows=1)
        save(p,[dict(row(1),answers=None)])
        with self.assertRaises(ValueError):prepare_course(p,sample,self.d/'bad',expected_rows=1)
        self.assertFalse((self.d/'bad').exists())

    def test_grpo_disjoint_deterministic_label_isolation_and_old_exclusion(self):
        a,c,s=self.sources();old=self.d/'old.json';save(old,[row(3)])
        first=select_grpo(a,c,s,self.d/'first',previous=[old])
        second=select_grpo(a,c,s,self.d/'second',previous=[old])
        self.assertEqual(first,second)
        rows=load(self.d/'first/questions.json');self.assertEqual(rows,load(self.d/'second/questions.json'))
        self.assertEqual(sum(r['split']=='dev' for r in rows),128)
        self.assertFalse({r['source_index'] for r in rows}&{0,1,2,3})
        self.assertTrue(all(set(r)=={'id','source_index','question','ctxs','split'} for r in rows))
        self.assertTrue(all(set(d)=={'title','text','score'} for r in rows for d in r['ctxs']))
        validate_new_selection('grpo',self.d/'first')
        rows[0]['question']='Changed after audit';save(self.d/'first/questions.json',rows)
        with self.assertRaises(ValueError):validate_new_selection('grpo',self.d/'first')

    def test_insufficient_grpo_data_does_not_write_partial_pack(self):
        a,c,s=self.sources();save(a/'2WikiMultiHopQA/train.json',[row(0),row(1)])
        with self.assertRaises(ValueError):select_grpo(a,c,s,self.d/'bad')
        self.assertFalse((self.d/'bad').exists())

    def test_new_grpo_prepare_uses_new_data_and_new_parent_provenance(self):
        a,c,s=self.sources();select_grpo(a,c,s,self.d/'selected')
        model=self.d/'model';save(model/'merge_manifest.json',dict(status='complete',changed_weight_elements={'w':10},
            adapter_sha256='a'*64,source_sha256='b'*64))
        cfg=dict(gpu_ids='4,5,6,7',sft=str(model),qwen='/qwen',inference_python='/infer/bin/python',training_python='/train/bin/python')
        with self.assertRaises(ValueError):prepare('grpo',cfg,self.d/'blocked',data_dir=self.d/'selected')
        out=prepare('grpo',cfg,self.d/'run',data_dir=self.d/'selected',accept_new_data=True)
        for name,h in load(out/'source_manifest.json')['files_sha256'].items():self.assertEqual(sha(out/name),h)
        self.assertEqual(load(out/'launch.json')['gpu_ids'],'4,5,6,7')
        self.assertIn('a'*64,(out/'model_provenance.py').read_text())
        self.assertTrue(load(out/'portability_changes.json')['changes']['new_selection_explicitly_accepted'])

    def test_unpack_flattens_only_required_data(self):
        archive=self.d/'author.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for name in NAMES:
                for split in ('train','test'):z.writestr(f'outer/dataset/{name}/{split}.json','[]')
            z.writestr('unused.txt','not copied')
        info=unpack(archive,self.d/'raw')
        self.assertEqual(len(info['files_sha256']),10);self.assertFalse((self.d/'raw/unused.txt').exists())
        with self.assertRaises(FileExistsError):unpack(archive,self.d/'raw')

    def test_unpack_rejects_ambiguous_or_missing_sources(self):
        archive=self.d/'bad.zip'
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('a/ASQA/train.json','[]');z.writestr('b/ASQA/train.json','[]')
        with self.assertRaises(ValueError):unpack(archive,self.d/'raw')
        self.assertFalse((self.d/'raw').exists())

    def test_defaults_use_new_training_outputs_not_private_adapters(self):
        c=configuration(ROOT/'configs/paths.example.json')
        for name in ('test','sample','baseline','sft','sft_adapter','sft_corpus','rl_adapter'):
            self.assertNotIn('private',Path(c[name]).relative_to(ROOT).parts)


if __name__=='__main__':unittest.main()
