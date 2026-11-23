import unittest

import torch

from train import Fusion, Policy


class FusionTests(unittest.TestCase):
    def test_tanh_changes_only_the_first_activation_of_each_branch(self):
        torch.set_num_threads(2)
        rgb = torch.randint(256, (2, 64, 64, 3), dtype=torch.uint8)
        for mode in ("add-deep", "concat-direct"):
            relu = Policy(mode, 3)
            tanh = Policy(mode, 3, first_activation="tanh")
            tanh.load_state_dict(relu.state_dict(), strict=True)
            for block in tanh.blocks:
                self.assertIsInstance(block.first_activation, torch.nn.Tanh)
            self.assertIsInstance(tanh.head[2], torch.nn.ReLU)
            self.assertFalse(torch.equal(relu(rgb), tanh(rgb)))

    def test_deeper_add_baseline(self):
        torch.set_num_threads(2)
        baseline = Policy("add-deep", 1)
        self.assertFalse(baseline.fusion)
        self.assertEqual([b.conv3 is not None for b in baseline.blocks], [False] * 5 + [True])
        self.assertEqual(baseline(torch.zeros(2, 64, 64, 3, dtype=torch.uint8)).shape, (2, 15))
        params = sum(p.numel() for p in baseline.parameters())
        self.assertEqual(params, 668015)
        for groups in (1, 2, 3, 6):
            count = sum(p.numel() for p in Policy("concat-direct", groups).parameters())
            self.assertLessEqual(abs(count - params) / params, 0.05)

    def test_direct_concat_has_no_additive_bypass(self):
        torch.set_num_threads(2)
        rgb = torch.randint(256, (2, 64, 64, 3), dtype=torch.uint8)
        baseline = sum(p.numel() for p in Policy("add", 1).parameters())
        for groups in (1, 2, 3, 6):
            model = Policy("concat-direct", groups)
            self.assertEqual(sum(p.numel() for p in model.parameters()), baseline + 3072 * groups)
            calls = []

            def record(module, args, output):
                self.assertEqual(args[0].shape[1], 64)
                self.assertEqual(output.shape[1], 32)
                calls.append(module)

            for mixer in model.fusion:
                mixer.register_forward_hook(record)
                torch.nn.init.zeros_(mixer[-1].weight)
            logits = model(rgb)
            self.assertEqual(calls, [model.fusion[i * groups // 6] for i in range(6)])
            torch.testing.assert_close(logits, model.head(torch.zeros(2, 32, 8, 8)))

    def test_matching_and_shared_groups(self):
        def size(model):
            return sum(p.numel() for p in model.parameters())

        baseline = size(Policy("add", 1))
        for groups in (1, 2, 3, 6):
            summed, concat = Policy("sum", groups), Policy("concat", groups)
            self.assertEqual(len(concat.fusion), groups)
            self.assertEqual(size(summed), size(concat))
            self.assertEqual(size(concat) - baseline, groups * 3 * 32**2)

    def test_identical_initial_policies(self):
        torch.set_num_threads(2)
        rgb = torch.randint(256, (2, 64, 64, 3), dtype=torch.uint8)
        outputs = []
        for mode in ("add", "sum", "concat"):
            torch.manual_seed(312)
            outputs.append(Policy(mode, 1)(rgb))
        for output in outputs[1:]:
            torch.testing.assert_close(output, outputs[0], rtol=0, atol=0)

    def test_concat_can_distinguish_equal_sums(self):
        summed, concat = Fusion("sum", 2), Fusion("concat", 2)
        with torch.no_grad():
            concat.up.weight.zero_()
            concat.up.weight[0, 0, 0, 0] = 1
            concat.down.weight[0, 0, 0, 0] = 1
        x, y = torch.ones(1, 2, 1, 1), torch.zeros(1, 2, 1, 1)
        torch.testing.assert_close(summed(x, y), summed(y, x))
        self.assertFalse(torch.equal(concat(x, y), concat(y, x)))
