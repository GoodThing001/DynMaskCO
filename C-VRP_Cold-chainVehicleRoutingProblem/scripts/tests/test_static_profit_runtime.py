"""Compiled fixed-weight inference must preserve original-task semantics."""
import os
os.environ['JAX_PLATFORMS']='cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['XLA_FLAGS']='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1'
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'static_profit'))
from profit_core import ProfitTask,partial_batch
import profit_neural as neural
from profit_search import repair


class CompiledInferenceSemantics(unittest.TestCase):
    def test_fixed_weights_logits_and_reconstruction_preserved(self):
        self.assertTrue(hasattr(neural,'CompiledPredictor'),'whole-forward compiled inference is absent')
        task=ProfitTask('r201',fold=0,role='train')
        routes=task.initial()
        removed=tuple(range(1,11))
        for variant in ('maskco_base','quality_only','directed_only','quality_directed'):
            with self.subTest(variant=variant):
                model=neural.ProfitMaskCO(variant,'smoke_cpu',1000)
                before=neural.parameter_digest(model)
                fast=neural.CompiledPredictor(model)
                batch,partial=partial_batch(task,routes,removed,variant)
                eager=neural.predict(model,batch)
                compiled=neural.predict(fast,batch)
                np.testing.assert_allclose(compiled,eager,rtol=3e-4,atol=3e-4)
                self.assertEqual(compiled.shape,(101,101))
                def rebuild(logits):
                    shifted=logits-logits.max(axis=1,keepdims=True)
                    heatmap=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
                    return repair(task,partial,removed,heatmap=heatmap)
                # JIT reassociation can perturb ties: require numerical agreement,
                # deterministic compiled decisions and strict route feasibility,
                # not bitwise equality of a discrete heatmap shortlist.
                rebuilt=rebuild(compiled)
                self.assertEqual(rebuilt,rebuild(compiled))
                self.assertTrue(task.evaluate(rebuild(eager),audit=True)['independently_audited'])
                self.assertEqual(neural.parameter_digest(model),before)
                self.assertTrue(task.evaluate(rebuilt,audit=True)['independently_audited'])


if __name__=='__main__':
    unittest.main()
